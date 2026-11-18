"""Normal-reference PCA scoring used in the same-cache control."""

import numpy as np


def matrix(value):
    if not isinstance(value, np.ndarray) or value.dtype not in (np.float32, np.float64):
        raise ValueError('Expected float32/float64 ndarray')
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError('Invalid feature matrix')
    return value.astype(np.float64)

def fit(reference, ev=0.99):
    x = matrix(reference)
    if len(x) < 2 or not 0 < ev <= 1: raise ValueError('Invalid reference/variance threshold')
    mu = x.mean(axis=0)
    x0 = x - mu
    values, vectors = np.linalg.eigh(x0.T @ x0 / (len(x) - 1))
    values, vectors = values[::-1], vectors[:, ::-1]
    total = values.sum()
    if not np.isfinite(total) or total <= 0: raise ValueError('Zero/nonfinite reference variance')
    # Author uses raw eigenvalues; no eigenvalue clipping or feature rescaling.
    k = min(len(values), int(np.searchsorted(np.cumsum(values) / total, ev)) + 1)
    return {'mu': mu, 'components': vectors[:, :k], 'k': k}

def residual(query, model):
    x = matrix(query)
    if x.shape[1] != len(model['mu']): raise ValueError('Feature dimension differs')
    z = x - model['mu']
    delta = z - (z @ model['components']) @ model['components'].T
    return np.sum(delta * delta, axis=1)

def fuse(layers):
    if layers.dtype != np.float32 or layers.ndim != 3 or layers.shape[0] != 3:
        raise ValueError('Expected three float32 layers')
    if not np.isfinite(layers).all(): raise ValueError('Nonfinite layers')
    mean = layers.mean(axis=0, dtype=np.float32)
    norm = np.linalg.norm(mean, axis=1, keepdims=True)
    if (norm <= 0).any(): raise ValueError('Zero fused norm')
    return (mean / norm).astype(np.float32)
