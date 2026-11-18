"""Scoring controls used by the full-image benchmark."""

from types import SimpleNamespace

import numpy as np

from aggregate import METHODS
from alf_reference import fuse as alf_fuse
from core import nearest_with_indices
from lnamd_scoring import NEW as LNAMD_MODES, descriptors, predict as lnamd_predict
import pca_scores
from weight_controls import normal_controls, score_controls


SUITES = {
    "main": METHODS,
    "weighting": ("reference_global", "reference_permuted"),
    "layers": ("layer4_identity", "layer8_identity", "mean44812_identity",
               "score_mean_48", "score_mean_412", "score_mean_812"),
    "scoring": ("alf_style_sharpness", "subspace_cached_pca", "musc_style_l2_r1"),
    "mask": ("masked_last", "masked_score_mean", "allpatch_last", "allpatch_score_mean"),
    "lnamd": LNAMD_MODES,
}


def fit_suite(suite, bank, instability, grid):
    if suite == "weighting":
        scales = np.median(instability.reshape(3, -1), axis=1)
        return (scales, *normal_controls(instability, scales))
    if suite == "scoring":
        return pca_scores.fit(pca_scores.fuse(bank), ev=.99)
    if suite == "layers":
        return pca_scores.fuse(bank)
    if suite == "lnamd":
        return descriptors(bank, grid, reference=True)
    return None


def control_maps(suite, bank, query, mask, grid, instability, fitted):
    if suite == "lnamd":
        def nearest(x, b):
            return nearest_with_indices(x, b)[0]

        maps, _ = lnamd_predict(SimpleNamespace(nearest_distances=nearest), query, fitted, mask, grid)
        return {key.removeprefix("distance_"): value for key, value in maps.items()}

    def restore(active):
        result = np.zeros(mask.size, dtype=np.float32)
        result[mask] = active
        return result.reshape(grid)

    if suite == "mask":
        unmasked = np.stack([nearest_with_indices(x, b)[0] for x, b in zip(query, bank)])
        last = unmasked[2].astype(np.float32).reshape(grid)
        equal = unmasked.mean(axis=0).astype(np.float32).reshape(grid)
        maps = (restore(last.ravel()[mask]), restore(equal.ravel()[mask]), last, equal)
        return dict(zip(SUITES[suite], maps))

    matches = [nearest_with_indices(x[mask], b) for x, b in zip(query, bank)]
    distances = np.stack([match[0] for match in matches]).astype(np.float32)
    if suite == "weighting":
        indices = np.stack([match[1] for match in matches])
        maps = np.stack([restore(d) for d in distances])
        return score_controls(maps, mask, grid, instability, fitted[0], indices, *fitted[1:])
    if suite == "scoring":
        full = np.stack([restore(d).ravel() for d in distances])
        alf = alf_fuse(full)[0].reshape(grid)
        pca = pca_scores.residual(pca_scores.fuse(query)[mask], fitted)
        l2 = np.sqrt(2 * distances.astype(np.float64)).mean(axis=0)
        return dict(zip(SUITES[suite], (alf, restore(pca), restore(l2))))

    feature_mean = nearest_with_indices(pca_scores.fuse(query)[mask], fitted)[0]
    pairs = [distances[list(pair)].mean(axis=0, dtype=np.float64)
             for pair in ((0, 1), (0, 2), (1, 2))]
    return {key: restore(values) for key, values in
            zip(SUITES[suite], (distances[0], distances[1], feature_mean, *pairs))}
