"""Recompute the main-table macro averages from the recorded official metrics."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parent
METHODS = ("last_identity", "score_mean", "reference_weighted")
METRICS = ("AUPRO_03", "P_AUROC", "P_F1max", "I_AUROC", "I_AP", "I_F1max")


def aggregate(dataset, run=None):
    manifest = json.loads((ROOT / "manifests" / f"{dataset}.json").read_text(encoding="utf-8"))
    categories = manifest["categories"]
    if run is None:
        with (ROOT / "results" / f"{dataset}.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
        for cat in categories:
            for group in ("0", "1", "2"):
                path = run / dataset / "original" / f"{cat}_{group}" / "metrics.json"
                metrics = json.loads(path.read_text(encoding="utf-8"))
                rows.extend(dict(category=cat, group=group, method=method, **values)
                            for method, values in metrics.items())
    keys = {(category, group, method)
            for category in categories for group in ("0", "1", "2") for method in METHODS}
    observed = {(row["category"], row["group"], row["method"]): row for row in rows}
    if len(rows) != len(keys) or observed.keys() != keys:
        raise ValueError(f"Incomplete or duplicate records for {dataset}")
    result = {}
    for method in METHODS:
        result[method] = {}
        for metric in METRICS:
            groups = [mean(float(observed[cat, group, method][metric]) for cat in categories)
                      for group in ("0", "1", "2")]
            result[method][metric] = {"mean": mean(groups), "sd": pstdev(groups)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Read newly evaluated metrics from this directory")
    parser.add_argument("--dataset", choices=("mvtec", "visa"))
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else ("mvtec", "visa")
    print(json.dumps({dataset: aggregate(dataset, args.run) for dataset in datasets}, indent=2))
