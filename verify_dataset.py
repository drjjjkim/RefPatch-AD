"""Check the referenced image and mask bytes against a benchmark manifest."""

import argparse
import json
from pathlib import Path

from reproduce import ROOT, checked_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifests" / f"{args.dataset}.json").read_text(encoding="utf-8"))
    counts = {"references": 0, "queries": 0, "masks": 0}
    for category in manifest["categories"].values():
        for reference in category["references"]:
            checked_file(args.data, reference["image"])
            counts["references"] += 1
        for query in category["queries"]:
            checked_file(args.data, query["image"])
            counts["queries"] += 1
            if query["mask"] is not None:
                checked_file(args.data, query["mask"])
                counts["masks"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
