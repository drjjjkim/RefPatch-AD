"""CPU numerical specification, NOT a MuSc+ reproduction or benchmark runner.

Input must precede per-patch L2 normalization. Implements the audited square-grid
LNAMD order with matching feature width: image-wide LN, zero-padded local mean,
then per-token L2. Not bitwise-equivalent to Torch Unfold/adaptive pooling.
No model, images, labels, detector metrics or files are read here.
"""
import numpy as np


def unit(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-10):
        raise ValueError('Zero/invalid aggregated token; no silent fallback')
    return x / norms


def ln_local(raw, grid, radius, *, provenance):
    """raw: views x patches x channels; views NEVER share normalization/pooling."""
    if provenance != 'pre_patch_l2':
        raise ValueError('Original pre-patch-L2 features required, not normalized cache')
    if (not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim != 3
            or not raw.size or not np.isfinite(raw).all()):
        raise ValueError('Finite float32 views x patches x channels required')
    if (len(grid) != 2 or any(type(v) is not int or v <= 0 for v in grid)
            or grid[0] != grid[1] or raw.shape[1] != grid[0]*grid[1]):
        raise ValueError('Audited official square-grid contract required')
    if type(radius) is not int or radius not in (1, 3, 5):
        raise ValueError('Fixed neighborhood widths 1/3/5 only')
    # Float64 reference arithmetic is intentional; GPU/Torch parity is separate.
    x = raw.astype(np.float64).reshape(raw.shape[0], *grid, raw.shape[-1])
    axes = (1, 2, 3)
    centered = x - x.mean(axis=axes, keepdims=True)
    x = centered / np.sqrt((centered**2).mean(axis=axes, keepdims=True) + 1e-5)
    h, w = grid
    pad = radius // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    pooled = np.zeros_like(x)
    for dy in range(radius):
        for dx in range(radius):
            pooled += padded[:, dy:dy+h, dx:dx+w, :]
    pooled /= radius**2
    return unit(pooled).reshape(raw.shape).astype(np.float32)
