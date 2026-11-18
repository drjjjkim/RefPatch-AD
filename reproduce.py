"""Run the original-resolution three-method benchmark from the fixed image lists."""

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from aggregate import METHODS, METRICS, ROOT
from controls import SUITES, control_maps, fit_suite
from core import reference_instability
from score_cached import score_maps


ANOMALYDINO = "b9d1c2648e3a5247437d4d953d907a8f3d994457"
DINOV2 = "85a24602099d397264d5b30461ad7f3bfd726ca1"
WEIGHT_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
EVALUATOR_SHA = {
    "utils.py": "8762ee1e6333f07720766381c064e0235440001e9b1889bd8d399984948b729b",
    "post_eval.py": "5ded024a6178a01815e4b37dfd812968fc605e0813546cfd00c41a63c840f7dd",
}
MVTEC_MASKED = {"capsule", "hazelnut", "pill", "screw", "toothbrush"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(root, record):
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != record["sha256"]:
        raise ValueError(f"Dataset file differs from the manifest: {path}")
    return path


def verify_checkout(path, commit):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise ValueError(f"Expected a clean checkout of {commit}: {path}")


def use_anomalydino(path):
    verify_checkout(path, ANOMALYDINO)
    for name, digest in EVALUATOR_SHA.items():
        if sha256(path / "src" / name) != digest:
            raise ValueError(f"Unexpected evaluator source: {name}")
    sys.path.insert(0, str(path.resolve()))


def build_encoder(args):
    verify_checkout(args.dinov2, DINOV2)
    if sha256(args.weights) != WEIGHT_SHA:
        raise ValueError("DINOv2-S/14 checkpoint SHA-256 mismatch")
    os.environ["XFORMERS_DISABLED"] = "1"
    import torch
    from src.backbones import get_model

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_hub = torch.hub.load

    def load_pinned(repo, name):
        if (repo, name) != ("facebookresearch/dinov2", "dinov2_vits14"):
            raise ValueError(f"Unexpected model request: {repo}, {name}")
        model = load_hub(str(args.dinov2.resolve()), name, source="local", pretrained=False)
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model

    with patch.object(torch.hub, "load", side_effect=load_pinned):
        wrapper = get_model("dinov2_vits14", args.device, smaller_edge_size=448)
    wrapper.model.eval().requires_grad_(False)
    return wrapper, torch


def extract(wrapper, torch, cv2, augment, root, record, *, reference=False, masking=False,
            pre_patch_l2=False):
    path = checked_file(root, record["image"])
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or list(image.shape[:2]) != [record["image"]["height"], record["image"]["width"]]:
        raise ValueError(f"Unexpected image size: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    views = augment(rgb) if reference else [rgb]
    if reference and (len(views) != 8 or not np.array_equal(views[0], rgb)):
        raise ValueError("Unexpected reference augmentation")
    pieces = []
    for view in views:
        tensor, grid = wrapper.prepare_image(view)
        if list(grid) != record["grid"]:
            raise ValueError(f"Unexpected patch grid: {path}")
        with torch.inference_mode():
            layers = wrapper.model.get_intermediate_layers(
                tensor.unsqueeze(0).to(wrapper.device), n=[3, 7, 11],
                norm=True, reshape=False, return_class_token=False,
            )
        raw = np.stack([layer.squeeze(0).cpu().numpy() for layer in layers])
        norms = np.linalg.norm(raw, axis=2, keepdims=True)
        if (raw.shape != (3, grid[0] * grid[1], 384) or raw.dtype != np.float32
                or not np.isfinite(raw).all() or np.any(norms <= 0)):
            raise ValueError(f"Invalid DINOv2 output: {path}")
        pieces.append(raw if pre_patch_l2 else raw / norms)
    features = np.concatenate(pieces, axis=1)
    active = np.ones(features.shape[1], dtype=bool)
    if masking:
        state = np.random.get_state()
        seed_bytes = ("official-query-pca-v1:" + record["image"]["sha256"]).encode()
        try:
            np.random.seed(int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16))
            active = wrapper.compute_background_mask(raw[-1], grid, threshold=10, masking_type=True)
        finally:
            np.random.set_state(state)
        if active.shape != (features.shape[1],) or active.dtype != np.bool_ or not active.any():
            raise ValueError(f"Invalid query mask: {path}")
    return features, active


def predict(args, selection, folder):
    import cv2
    from src.utils import augment_image

    cv2.setNumThreads(args.threads)
    wrapper, torch = build_encoder(args)
    raw = args.suite == "lnamd"
    bank, _ = extract(wrapper, torch, cv2, augment_image, args.data, selection["reference"],
                      reference=True, pre_patch_l2=raw)
    instability = reference_instability(bank) if args.suite in ("main", "weighting") else None
    fitted = fit_suite(args.suite, bank, instability, selection["reference"]["grid"])
    masking = args.dataset == "visa" or args.category in MVTEC_MASKED
    folder.mkdir(parents=True)
    for query in selection["queries"]:
        features, active = extract(wrapper, torch, cv2, augment_image, args.data, query,
                                   masking=masking, pre_patch_l2=raw)
        if args.suite == "main":
            maps = score_maps(bank, features, active, query["grid"], instability)
        else:
            controls = control_maps(args.suite, bank, features, active, query["grid"], instability, fitted)
            maps = {f"distance_{key}": value for key, value in controls.items()}
            maps["query_mask"] = active
        with (folder / f'{query["id"]}.npz').open("xb") as stream:
            np.savez_compressed(stream, **maps)
        print(f'PREDICTED {query["id"]}', flush=True)
    (folder / "inputs.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


def evaluate(args, selection, folder, methods=METHODS):
    import cv2
    import tifffile
    from src import post_eval
    from src.utils import dists2map

    cv2.setNumThreads(args.threads)
    if json.loads((folder / "inputs.json").read_text(encoding="utf-8")) != selection:
        raise ValueError("Evaluation requires the complete prediction set for this reference")
    queries = selection["queries"]
    ground_truth = [str(checked_file(args.data, q["mask"])) if q["mask"] else None for q in queries]
    if not any(ground_truth) or all(ground_truth):
        raise ValueError("Official evaluation requires normal and anomalous test images")
    for q in queries:
        checked_file(args.data, q["image"])
    results = {}
    for method in methods:
        with TemporaryDirectory(prefix="maps_", dir=folder) as scratch:
            prefixes = []
            for query in queries:
                qid = query["id"]
                with np.load(folder / f"{qid}.npz", allow_pickle=False) as archive:
                    scores = archive[f"distance_{method}"]
                if (scores.shape != tuple(query["grid"]) or scores.dtype != np.float32
                        or not np.isfinite(scores).all() or (scores < 0).any()):
                    raise ValueError(f"Invalid scores: {qid}, {method}")
                prefix = Path(scratch) / qid
                np.save(prefix.with_suffix(".npy"), scores, allow_pickle=False)
                size = (query["image"]["height"], query["image"]["width"])
                full = dists2map(scores, (*size, 3))
                if full.shape != size or not np.isfinite(full).all():
                    raise ValueError(f"Invalid original-resolution map: {qid}")
                tifffile.imwrite(prefix.with_suffix(".tiff"), full)
                prefixes.append(str(prefix))
            auc, ap, f1 = post_eval.eval_classification(
                ground_truth, prefixes, aggregation_statistics="meantop1p",
            )
            # The upstream extension search finds the same TIFF twice on Windows.
            tiff_reader = (patch.object(post_eval, "read_tiff", partial(post_eval.read_tiff, exts=(".tiff",)))
                           if os.name == "nt" else nullcontext())
            with tiff_reader:
                pro, pixel_auc, pixel_f1 = post_eval.eval_segmentation(
                    ground_truth, prefixes, pro_integration_limit=.3, delete_tiff_files=False,
                )
            values = tuple(map(float, (pro, pixel_auc, pixel_f1, auc, ap, f1)))
            if not all(np.isfinite(v) and -1e-12 <= v <= 1 + 1e-12 for v in values):
                raise ValueError(f"Invalid official metrics: {method}")
            results[method] = dict(zip(METRICS, values))
    with (folder / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "evaluate"))
    parser.add_argument("--dataset", required=True, choices=("mvtec", "visa"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--group", required=True, type=int, choices=range(3))
    parser.add_argument("--cohort", choices=("original", "additional"), default="original")
    parser.add_argument("--suite", choices=SUITES, default="main")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--anomalydino", required=True, type=Path)
    parser.add_argument("--dinov2", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--query", help="Optional single query for prediction checks")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.phase == "predict" and (args.dinov2 is None or args.weights is None):
        parser.error("predict requires --dinov2 and --weights")
    if args.phase == "evaluate" and args.query:
        parser.error("evaluate requires all queries for the category")
    if args.suite == "mask" and (args.cohort != "original" or args.group != 0):
        parser.error("the paper's mask control uses original reference group 0")
    if args.suite == "lnamd" and (args.dataset != "mvtec" or args.cohort != "original"):
        parser.error("the paper's LNAMD control uses original MVTec AD references")
    manifest = json.loads((ROOT / "manifests" / f"{args.dataset}.json").read_text(encoding="utf-8"))
    category = manifest["categories"].get(args.category)
    if category is None:
        parser.error("unknown category for the selected dataset")
    queries = [q for q in category["queries"] if not args.query or q["id"] == args.query]
    if not queries:
        parser.error("unknown query ID for the selected category")
    reference = next(r for r in category["references"]
                     if (r["cohort"], r["group"]) == (args.cohort, args.group))
    selection = {"reference": reference, "queries": queries}
    suffix = f"_{args.query}" if args.query else ""
    folder = args.out / args.dataset / args.cohort / f"{args.category}_{args.group}{suffix}"
    if args.suite != "main":
        selection["suite"] = args.suite
        folder = folder.with_name(f"{folder.name}_{args.suite}")
    use_anomalydino(args.anomalydino)
    if args.phase == "predict":
        predict(args, selection, folder)
    else:
        evaluate(args, selection, folder, SUITES[args.suite])


if __name__ == "__main__":
    main()
