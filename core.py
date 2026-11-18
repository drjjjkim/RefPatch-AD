"""Exploratory scoring core. No dataset I/O, labels, files or automatic evaluation."""
import numpy as np


def checked_unit(x):
    a = np.asarray(x)
    if (a.ndim != 2 or not a.shape[0] or not a.shape[1]
            or not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all()):
        raise ValueError('nonempty finite floating matrix required')
    if np.max(np.abs(np.linalg.norm(a, axis=1) - 1)) > 1e-4:
        raise ValueError('cached features must already be unit-normalized')
    return np.array(a, dtype=np.float64, copy=True)


def nearest_with_indices(query, bank, chunk=64):
    """Match the existing half squared-L2 implementation, plus first argmin."""
    q, b = checked_unit(query), checked_unit(bank)
    if q.shape[1] != b.shape[1] or type(chunk) is not int or chunk < 1:
        raise ValueError('invalid shape or chunk')
    norms = np.einsum('ij,ij->i', b, b)
    values, indices = [], []
    for start in range(0, len(q), chunk):
        part = q[start:start + chunk]
        d = (np.einsum('ij,ij->i', part, part)[:, None] + norms[None] - 2 * part @ b.T) / 2
        d = np.maximum(0, d)
        idx = d.argmin(axis=1)
        indices.append(idx)
        values.append(d[np.arange(len(part)), idx])
    return np.concatenate(values), np.concatenate(indices)


def reference_instability(layers, views=8, chunk=64, progress=None):
    """[L,V*N,D] -> [L,V,N], nearest distance per other view, then median.

    All cached reference patches, including padding, are retained as in baseline.
    No geometry correspondence is assumed; independent views are not claimed.
    """
    a = np.asarray(layers)
    if (a.ndim != 3 or a.shape[0] < 1 or type(views) is not int or views < 2
            or a.shape[1] < views or a.shape[1] % views):
        raise ValueError('invalid [layer, view*patch, channel] array')
    n = a.shape[1] // views
    out = np.empty((a.shape[0], views, n), dtype=np.float64)
    for layer in range(a.shape[0]):
        checked_unit(a[layer])
        blocks = a[layer].reshape(views, n, a.shape[-1])
        for view in range(views):
            costs = [nearest_with_indices(blocks[view], blocks[v], chunk)[0]
                     for v in range(views) if v != view]
            out[layer, view] = np.median(costs, axis=0)
            if progress is not None:
                progress(layer, view)
    return out


def score(query, banks, mask, instability, chunk=64, blend=0.5, eps=1e-6):
    """Return equal and adaptive float32 maps, weights and indices.

    Inputs: query [L,N,D], banks [L,V*Nref,D], bool mask [N], instability
    [L,V,Nref]. Each distance is cast float32 before float64 equal averaging,
    matching the prior score_mean numerical path. Only active query patches
    are matched; masked cells are exactly zero in both outputs.
    """
    q, b, m, u = map(np.asarray, (query, banks, mask, instability))
    if (q.ndim != 3 or b.ndim != 3 or q.shape[0] != b.shape[0]
            or q.shape[2] != b.shape[2] or m.dtype != np.bool_ or m.shape != (q.shape[1],)
            or not m.any() or u.ndim != 3 or u.shape[0] != b.shape[0]
            or u.shape[1] * u.shape[2] != b.shape[1]):
        raise ValueError('shape or mask mismatch')
    if not np.isfinite(u).all() or np.any(u < 0) or not (0 <= blend <= 1) or not (eps > 0):
        raise ValueError('invalid calibration')
    flats = u.reshape(u.shape[0], -1)
    scale = np.median(flats, axis=1)
    if not np.isfinite(scale).all() or np.any(scale <= eps):
        raise ValueError('degenerate scale, do not silently fallback')
    ds, indices = [], []
    for layer in range(q.shape[0]):
        v, j = nearest_with_indices(q[layer, m], b[layer], chunk)
        ds.append(v.astype(np.float32)); indices.append(j)
    ds, indices = np.stack(ds), np.stack(indices)
    selected = np.take_along_axis(flats, indices, axis=1)
    raw = 1 / (eps + selected / scale[:, None])
    adaptive = raw / raw.sum(axis=0, keepdims=True)
    weights = (1 - blend) / q.shape[0] + blend * adaptive
    equal = np.zeros(len(m), dtype=np.float32)
    fused = np.zeros(len(m), dtype=np.float32)
    equal[m] = ds.astype(np.float64).mean(axis=0).astype(np.float32)
    fused[m] = (ds.astype(np.float64) * weights).sum(axis=0).astype(np.float32)
    if not np.isfinite(fused).all():
        raise ValueError('nonfinite output')
    return equal, fused, weights, indices
