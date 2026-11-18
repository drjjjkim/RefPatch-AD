"""Controls from saved tables, indices and layer distances. No NN or model code."""
import numpy as np

EPS=1e-6
BLEND=.5
SEED=20261102
MODES=('reference_global','reference_permuted')

def reference_table(instability,scales):
    u,s=np.asarray(instability),np.asarray(scales)
    if (u.dtype!=np.float64 or u.ndim!=3 or u.shape[:2]!=(3,8) or u.shape[2]<1
        or not np.isfinite(u).all() or np.any(u<0) or s.dtype!=np.float64 or s.shape!=(3,)
        or not np.isfinite(s).all() or np.any(s<=EPS)):
        raise ValueError('Invalid original normal table/scales')
    flat=u.reshape(3,-1)
    if not np.array_equal(np.median(flat,axis=1),s):raise ValueError('Original median scales differ')
    return flat,s

def local_weights(flat,scales,indices):
    j=np.asarray(indices)
    if (flat.ndim!=2 or flat.shape[0]!=3 or j.ndim!=2 or j.shape[0]!=3
        or not np.issubdtype(j.dtype,np.integer) or not j.shape[1]
        or np.any(j<0) or np.any(j>=flat.shape[1])):raise ValueError('Invalid saved nearest indices')
    selected=np.take_along_axis(flat,j,axis=1)
    raw=1/(EPS+selected/scales[:,None])
    return (1-BLEND)/3 + BLEND*(raw/raw.sum(axis=0,keepdims=True))

def normal_controls(instability,scales):
    flat,s=reference_table(instability,scales)
    raw=1/(EPS+flat/s[:,None])
    normalized=raw/raw.sum(axis=0,keepdims=True)
    global_weights=(1-BLEND)/3+BLEND*normalized.mean(axis=1)
    permutation=np.random.default_rng(SEED).permutation(flat.shape[1])
    if (not np.isfinite(global_weights).all() or not np.isclose(global_weights.sum(),1,rtol=0,atol=1e-12)):
        raise ValueError('Invalid normal-only global weights')
    return global_weights,permutation

def validate_distances(distances,mask,grid):
    d,m=np.asarray(distances),np.asarray(mask)
    if (len(grid)!=2 or any(type(v)is not int or v<1 for v in grid)
        or d.dtype!=np.float32 or d.shape!=(3,*grid) or not np.isfinite(d).all() or np.any(d<0)
        or m.dtype!=np.bool_ or m.shape!=(int(np.prod(grid)),) or not m.any()
        or np.any(d.reshape(3,-1)[:,~m]!=0)):
        raise ValueError('Invalid saved layer distances or original mask')
    return d,m

def restore(active,mask,grid):
    out=np.zeros(mask.size,dtype=np.float32);out[mask]=active.astype(np.float32)
    return out.reshape(grid)

def verify_originals(distances,mask,grid,instability,scales,indices,saved_weights,equal,local):
    d,m=validate_distances(distances,mask,grid);flat,s=reference_table(instability,scales)
    w=local_weights(flat,s,indices)
    if w.shape!=(3,int(m.sum())) or np.asarray(saved_weights).dtype!=np.float64 or not np.array_equal(w,saved_weights):
        raise ValueError('Saved local weights cannot be exactly reconstructed')
    active=d.reshape(3,-1)[:,m].astype(np.float64)
    eq=restore(active.mean(axis=0),m,grid)
    loc=restore((active*w).sum(axis=0),m,grid)
    for rebuilt,old,name in ((eq,equal,'equal'),(loc,local,'local')):
        if (old.dtype!=np.float32 or old.shape!=tuple(grid) or not np.isfinite(old).all()
            or np.any(old<0) or not np.array_equal(rebuilt,old)):
            raise ValueError('Original '+name+' map cannot be reconstructed bitwise; no tolerance changes')
    return active,w

def score_controls(distances,mask,grid,instability,scales,indices,global_weights,permutation):
    d,m=validate_distances(distances,mask,grid);flat,s=reference_table(instability,scales)
    expected_global,expected_perm=normal_controls(instability,scales)
    if not np.array_equal(global_weights,expected_global) or not np.array_equal(permutation,expected_perm):
        raise ValueError('Controls differ from fixed normal-only rule/seed')
    if np.asarray(indices).shape!=(3,int(m.sum())):raise ValueError('Saved index/mask shape differs')
    shuffled=local_weights(flat[:,permutation],s,indices)
    active=d.reshape(3,-1)[:,m].astype(np.float64)
    result=dict(reference_global=restore((active*global_weights[:,None]).sum(axis=0),m,grid),
        reference_permuted=restore((active*shuffled).sum(axis=0),m,grid))
    if any(not np.isfinite(v).all() or np.any(v<0) for v in result.values()):raise ValueError('Invalid new control maps')
    return result
