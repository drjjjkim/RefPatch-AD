"""Same-protocol LNAMD core adaptation, NOT full MuSc+; no Torch or GT."""
import numpy as np
from ln_core import ln_local

NEW=('lnamd_l2_r1','lnamd_l2_r135')
REUSED=('score_mean','musc_style_l2_r1')
ALL=REUSED+NEW


def descriptors(raw,grid,reference=False):
    n=int(np.prod(grid));views=8 if reference else 1
    if (raw.dtype!=np.float32 or raw.shape!=(3,views*n,384) or not np.isfinite(raw).all()
            or len(grid)!=2 or any(type(v)is not int or v<1 for v in grid) or grid[0]!=grid[1]):
        raise ValueError('Fixed raw3 layers/full square grid/eight reference views required')
    # Each rotation view independently normalized/pooled, never along bank axis.
    return np.stack([np.stack([ln_local(layer.reshape(views,n,384),grid,r,
        provenance='pre_patch_l2').reshape(views*n,384) for layer in raw]) for r in (1,3,5)])


def fuse(half_maps,mask,grid):
    if (half_maps.dtype!=np.float32 or half_maps.shape!=(3,3,*grid)
            or not np.isfinite(half_maps).all() or np.any(half_maps<0)
            or mask.dtype!=np.bool_ or mask.shape!=(int(np.prod(grid)),) or not mask.any()
            or np.any(half_maps.reshape(9,-1)[:,~mask]!=0)):
        raise ValueError('Nine restored half-squared-L2 maps and exact original mask required')
    # Match the old L2-r1 cache precision: quantize half distance to float32
    # BEFORE float64 sqrt and mean. This is not direct Torch cdist bitwise parity.
    d=np.sqrt(2*half_maps.astype(np.float64))
    return {'distance_'+NEW[0]:d[0].mean(axis=0).astype(np.float32),
            'distance_'+NEW[1]:d.mean(axis=(0,1)).astype(np.float32)}


def predict(core,raw,bank,mask,grid):
    n=int(np.prod(grid))
    if (mask.dtype!=np.bool_ or mask.shape!=(n,) or not mask.any()
            or bank.dtype!=np.float32 or bank.ndim!=4 or bank.shape[:2]!=(3,3)
            or bank.shape[-1]!=384 or bank.shape[2]<8 or bank.shape[2]%8!=0 or not np.isfinite(bank).all()):
        raise ValueError('Full original reference bank/mask required')
    query=descriptors(raw,grid)
    maps=np.zeros((3,3,*grid),dtype=np.float32)
    for ri in range(3):
        for li in range(3):
            active=np.asarray(core.nearest_distances(query[ri,li,mask],bank[ri,li]),dtype=np.float32)
            if active.shape!=(int(mask.sum()),) or not np.isfinite(active).all() or np.any(active<0):
                raise ValueError('Invalid fixed exact-NN output')
            maps[ri,li].reshape(-1)[mask]=active
    return fuse(maps,mask,grid),maps
