"""Float64 mathematical reference for pinned ALF sharpness core, not full ALF.

Source: Deyu008/ALF commit f17802a0acac5dab0dd1e2f20e32f81a0ce546be,
adaptive_r_fusion.py:478-487,511-540,1393-1492; core-e01 YAML.
No Torch numerical parity or anomaly performance is implied.
"""
import numpy as np

PARAMETERS = dict(eps=1e-8, temperature=0.2, percentile=95.0,
                  floor=0.02, ceil=0.7, shrink=0.1)


def fuse(maps):
    """One query, full masked grid, exactly three layers. No active-only topk."""
    if not isinstance(maps, np.ndarray) or maps.dtype != np.float32:
        raise ValueError('Expected float32 array')
    if maps.ndim != 2 or maps.shape[0] != 3 or maps.shape[1] < 1:
        raise ValueError('Expected (3,P), P>=1')
    if not np.isfinite(maps).all() or np.any(maps < 0):
        raise ValueError('Invalid distances')
    x = maps.astype(np.float64)
    norm = (x - x.min(axis=1, keepdims=True)) / (
        x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-8)
    k = max(1, int(x.shape[1] * (100 - 95.0) / 100))
    tail = np.sort(norm, axis=1)[:, -k:]
    sd = np.zeros(3) if k == 1 else tail.std(axis=1, ddof=1)
    scores = sd / (tail.mean(axis=1) + 1e-8)
    logits = scores / 0.2
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    weights = np.clip(weights, 0.02, 0.7)
    weights /= weights.sum()
    weights = 0.9 * weights + 0.1 / 3
    weights = np.clip(weights, 0.02, 0.7)
    weights /= weights.sum()
    result = (x * weights[:, None]).sum(axis=0).astype(np.float32)
    return result, weights, scores
