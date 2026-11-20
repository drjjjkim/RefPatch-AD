# RefPatch-AD

RefPatch-AD is a training-free method for 1-shot industrial anomaly localization. The repository provides the three benchmark scoring methods, fixed dataset manifests, and the records underlying the main results table.

## Quick start

Use Python 3.10–3.12. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python aggregate.py
```

The test generates unit-normalized synthetic features and runs the scoring command with a partially masked query. The aggregation command reads the recorded official metrics in `results/`, averages categories within each of the three original reference groups, then reports the mean and population standard deviation across the three groups. Metrics are stored on the 0–1 scale. This command reproduces table arithmetic from the recorded metrics.

## Score a cached query

```sh
python score_cached.py --reference bank.npz --query query.npz --out prediction.npz
```

The two input NPZ archives use the experiment cache format:

| Field | Reference | Query |
|---|---|---|
| `layer_features` | float32 `[3, 8*H*W, 384]`, view-major within each layer | float32 `[3, H*W, 384]` |
| `grid` | `[H,W]` | `[H,W]` |
| `layers_one_based` | `[4,8,12]` | `[4,8,12]` |
| `angles` | `[0,45,90,135,180,225,270,315]` | `[0]` |
| `query_mask` | Boolean vector with all positions active | Boolean vector in query grid order |

Features are L2-normalized per patch. The command writes three float32 patch-grid maps, `distance_last_identity`, `distance_score_mean`, and `distance_reference_weighted`. Inactive positions are zero. The command calls the reference fitting and weighted scorer from the paper experiments in `core.py`. All three scores use half squared Euclidean distances. See [Verification](#verification) for the real-image comparisons and their scope.

## Datasets and experiment records

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), 15 categories and 1,725 test images. Manifest paths start at the extracted dataset root.
- [VisA](https://github.com/amazon-science/spot-diff), 12 categories and 2,162 test images. Use the author's one-class split in the prepared `1cls` layout. Manifest paths start at that directory.

`manifests/` contains dataset-relative image and mask paths, SHA-256 values, grids, and the fixed original and additional normal reference identities. Each reference represents its own 1-shot run. A null mask means a normal test image. The feature archive hashes identify the original caches and record provenance.

`results/` contains 243 original-reference metric rows across three methods. The equal and weighted records come from the full MVTec AD and VisA reference-weighting evaluations. The last-layer records come from the same-protocol results retained by the MVTec AD and VisA PCA-control reports. The equal-weight records agree exactly across those two sets of reports. Each row has six official metrics.

Raw dataset images, weights and feature caches are obtained or prepared separately. Dataset access follows each provider's terms.

### Prepare VisA

Download `VisA_20220922.tar` from the [official VisA project](https://github.com/amazon-science/spot-diff#data-download). Use the split CSV inside that archive with the pinned official preparation script. The commands below assume a new `data/VisA_pytorch` directory.

```sh
python -m pip install -r requirements-evaluation.txt
mkdir -p data/VisA_raw
tar -xf VisA_20220922.tar -C data/VisA_raw
git clone https://github.com/amazon-science/spot-diff.git external/spot-diff
git -C external/spot-diff checkout --detach 2a692ab575001cbde74d402d897a7286086c6199
python external/spot-diff/utils/prepare_data.py --split-type 1cls --data-folder data/VisA_raw --save-folder data/VisA_pytorch --split-file data/VisA_raw/split_csv/1cls.csv
python verify_dataset.py --dataset visa --data data/VisA_pytorch/1cls
```

The split CSV SHA-256 is `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`. The preparation command was run on the official archive with Pillow 9.5.0. The manifest check matched all 72 normal reference images (two cohorts), 2,162 test images, and 1,200 binary masks byte for byte. Use `data/VisA_pytorch/1cls` as `--data` in the VisA experiments.

## Run the original-resolution benchmark

The benchmark uses [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) for image transforms, masks and evaluation, and [DINOv2](https://github.com/facebookresearch/dinov2) for patch embeddings. Prepare clean checkouts:

```sh
git -c core.autocrlf=false clone https://github.com/dammsi/AnomalyDINO.git external/AnomalyDINO
git -C external/AnomalyDINO checkout --detach b9d1c2648e3a5247437d4d953d907a8f3d994457
git -c core.autocrlf=false clone https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout --detach 85a24602099d397264d5b30461ad7f3bfd726ca1
mkdir -p cache
curl -fL https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -o cache/dinov2_vits14_pretrain.pth
python -m pip install -r requirements-evaluation.txt
```

For prediction, also install a compatible PyTorch and TorchVision pair for your hardware. The paper's prediction environment reported PyTorch 2.14.0+cu130 and TorchVision 0.29.0. CPU and CUDA are selectable. Original measurements used CUDA for feature extraction and CPU for scoring and evaluation.

With the paths below set for your installation, run one category and one reference group:

```sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
DATA=/path/to/mvtec_anomaly_detection
OUT=outputs/benchmark
python reproduce.py predict --dataset mvtec --category bottle --group 0 --data "$DATA" --out "$OUT" --anomalydino external/AnomalyDINO --dinov2 external/dinov2 --weights cache/dinov2_vits14_pretrain.pth --device cuda
python reproduce.py evaluate --dataset mvtec --category bottle --group 0 --data "$DATA" --out "$OUT" --anomalydino external/AnomalyDINO
```

Append `--query q063` to the prediction command for a one-image check. Evaluation takes the full category prediction set. For VisA use `--dataset visa` and set `DATA` to the prepared `1cls` directory. Run each dataset's manifest categories with groups 0, 1 and 2, then run `python aggregate.py --run "$OUT"` to compute the full table. Each category–reference pair has its own output directory. The official exact pixel metrics sort original-resolution pixels, so schedule one evaluation process at a time according to available RAM and temporary disk space.

Patch predictions are retained. Original-resolution TIFFs and evaluator NPYs are written to a per-method temporary directory. Evaluation uses the original ground-truth masks, the full-grid mean of the top 1% for image scores, and AUPRO at a 0.30 false positive rate limit. The image workflow regenerates query masks from the pinned PCA implementation and the recorded image-hash seed.

### Verification

For MVTec AD bottle, original reference 0 and query `q063`, CPU extraction from the two original images reproduced the archived query mask exactly. Maximum absolute differences against maps scored from the archived features were `6.523e-6` for last-layer matching, `2.615e-6` for equal fusion, and `2.515e-6` for RefPatch-AD. This category uses the all-patch query policy.

For MVTec AD toothbrush, original reference 0, all 42 PCA query masks matched the archived masks exactly. The end-to-end CPU run computed all six official metrics for each method. The nine image-level metrics matched the records in `results/mvtec.csv` exactly. Across all 18 metrics, the largest absolute difference was `5.40e-5` percentage points. These comparisons cover the named cases. The two full-dataset tables are recomputed from the recorded metrics with `python aggregate.py`.

## Supporting experiments

See [Supporting experiments](SUPPORTING_EXPERIMENTS.md) for the paper-to-code mapping, control settings, and the boundaries of the verification. Use `python supporting_results.py` to reaggregate the recorded control metrics and timing observations and recalculate the category bootstrap.

## License

The original RefPatch-AD code is licensed under the [Apache License, Version 2.0](LICENSE). Third-party code, datasets, and model weights are governed by their respective terms.
