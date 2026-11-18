"""Score a query from the feature archive schema used in the experiments."""

import argparse
from pathlib import Path

import numpy as np

from core import nearest_with_indices, reference_instability, score


def read_cache(path, reference=False):
    angles = list(range(0, 360, 45)) if reference else [0]
    with np.load(path, allow_pickle=False) as archive:
        features = archive["layer_features"]
        mask = archive["query_mask"]
        grid = archive["grid"].tolist()
        if (archive["layers_one_based"].tolist() != [4, 8, 12]
                or archive["angles"].tolist() != angles):
            raise ValueError(f"Unexpected layer or view order in {path}")
    if len(grid) != 2 or any(type(size) is not int or size < 1 for size in grid):
        raise ValueError(f"Invalid grid in {path}")
    patches = grid[0] * grid[1] * len(angles)
    if (features.dtype != np.float32 or features.shape != (3, patches, 384)
            or not np.isfinite(features).all() or mask.dtype != np.bool_
            or mask.shape != (patches,) or not mask.any()
            or (reference and not mask.all())
            or np.max(np.abs(np.linalg.norm(features, axis=2) - 1)) > 1e-4):
        raise ValueError(f"Invalid unit features or mask in {path}")
    return features, mask, grid


def score_maps(banks, features, mask, grid, instability):
    equal, weighted, _, _ = score(features, banks, mask, instability)
    last = np.zeros(mask.size, dtype=np.float32)
    last[mask] = nearest_with_indices(features[2, mask], banks[2])[0].astype(np.float32)
    return dict(
        distance_last_identity=last.reshape(grid),
        distance_score_mean=equal.reshape(grid),
        distance_reference_weighted=weighted.reshape(grid),
        query_mask=mask,
    )


def score_archives(reference, query, output):
    banks, _, _ = read_cache(reference, reference=True)
    features, mask, grid = read_cache(query)
    maps = score_maps(banks, features, mask, grid, reference_instability(banks))
    with output.open("xb") as stream:
        np.savez_compressed(stream, **maps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score_archives(args.reference, args.query, args.out)


if __name__ == "__main__":
    main()
