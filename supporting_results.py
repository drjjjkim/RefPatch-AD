"""Summarize recorded control metrics and paired timing observations."""

import csv
import json
from statistics import mean, median, pstdev

import numpy as np

from aggregate import METRICS, ROOT


DATA = ROOT / "results/supporting"


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def metric_index():
    index = {}

    def insert(key, row):
        if key in index:
            raise ValueError(f"Duplicate metric row: {key}")
        values = {metric: float(row[metric]) for metric in METRICS}
        if not all(-1e-12 <= v <= 1 + 1e-12 for v in values.values()):
            raise ValueError(f"Invalid metric row: {key}")
        index[key] = values

    for dataset in ("mvtec", "visa"):
        with (ROOT / "results" / f"{dataset}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                insert(("original", dataset, row["category"], row["group"], row["method"]), row)
    with (DATA / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            insert(tuple(row[k] for k in ("experiment", "dataset", "category", "group", "method")), row)
    return index


def category_names(dataset):
    manifest = ROOT / "manifests" / f"{dataset}.json"
    return list(json.loads(manifest.read_text(encoding="utf-8"))["categories"])


def summarize(index):
    results = {}
    for experiment, dataset, method in sorted({(e, d, m) for e, d, _, _, m in index}):
        groups = ("0",) if experiment == "mask" else ("0", "1", "2")
        categories = category_names(dataset)
        output = results.setdefault(experiment, {}).setdefault(dataset, {})
        output[method] = {}
        for metric in METRICS:
            scores = [mean(index[experiment, dataset, cat, group, method][metric]
                           for cat in categories) for group in groups]
            output[method][metric] = {"mean": mean(scores), "sd": pstdev(scores)}
    return results


def bootstrap_aupro(index):
    result = {}
    for i, (cohort, dataset) in enumerate((c, d) for c in ("original", "additional")
                                         for d in ("mvtec", "visa")):
        categories = category_names(dataset)
        paired = np.array([
            [100 * (index[cohort, dataset, cat, str(g), "reference_weighted"]["AUPRO_03"]
                    - index[cohort, dataset, cat, str(g), "score_mean"]["AUPRO_03"])
             for g in range(3)] for cat in categories
        ]).mean(axis=1)
        n = len(paired)
        sampled = np.random.default_rng(20261104 + i).integers(0, n, size=(100000, n))
        ci = np.quantile(paired[sampled].mean(axis=1), [.025, .975], method="linear")
        leave_one = (paired.sum() - paired) / (n - 1)
        result.setdefault(cohort, {})[dataset] = {
            "mean_delta_pp": float(paired.mean()),
            "category_bootstrap_percentile95_pp": ci.tolist(),
            "leave_one_category_out_range_pp": [float(leave_one.min()), float(leave_one.max())],
        }
    return result


def timing():
    online = read_json("online_timing.json")
    cpu = read_json("cpu_timing.json")
    result = {}
    for dataset in ("mvtec", "visa"):
        item = result[dataset] = {
            method: median(r["online_cached_mask_ms"] for r in online
                           if r["dataset"].lower() == dataset and r["mode"] == method)
            for method in ("last_identity", "score_mean")
        }
        fit, delta = [], []
        for sample in (s for s in cpu if s["dataset"] == dataset):
            rounds = {}
            for row in sample["measurements"]:
                if row["stage"] == "normal_fit":
                    fit.append(row["ms"])
                elif row["stage"] == "post":
                    rounds.setdefault(row["round"], {})[row["mode"]] = row["ms"]
            delta.extend(v["reference_weighted"] - v["score_mean"] for v in rounds.values())
        item["reference_fit_seconds"] = median(fit) / 1000
        item["post_paired_delta_ms"] = median(delta)
    return result


def regional_response():
    diagnostics = read_json("response_category_records.json")["diagnostics"]
    result = {}
    for dataset, kinds in diagnostics.items():
        for kind, entry in kinds.items():
            values = {}
            for field in entry["macro"]:
                group_means = []
                for group in ("0", "1", "2"):
                    valid = [r[group][field]["mean"] for r in entry["categories"].values()
                             if r[group][field]["mean"] is not None]
                    if valid:
                        group_means.append(mean(valid))
                values[field] = mean(group_means) if group_means else None
            result.setdefault(dataset, {})[kind] = values
    return result


if __name__ == "__main__":
    index = metric_index()
    print(json.dumps({
        "metrics": summarize(index),
        "fixed_layer_aggregates_percent": read_json("layer_aggregates_percent.json"),
        "bootstrap_aupro": bootstrap_aupro(index),
        "timing": timing(),
        "regional_response": regional_response(),
    }, indent=2))
