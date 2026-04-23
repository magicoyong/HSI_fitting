"""
Endmember initialization via NMF.

Provides two entry points:
  1. masked_nmf_initialization — uses only masked (observed) HSI pixels.
     Missing positions are filled differently depending on mask type:
       - elementwise: per-pixel spectral mean (observed bands → missing bands);
         fully-missing pixels fall back to local spatial same-band mean.
       - random / pixel-wise: local spatial same-band mean from neighbouring
         observed pixels (expanding window); global per-band mean only as
         last-resort fallback.
     This is the ONLY path that should be used for HSI inpainting.
  2. (legacy) nmf_initialization — uses full GT HSI.  DEPRECATED for
     inpainting because it leaks test information.
"""

import argparse
import numpy as np
import scipy.io
from sklearn.decomposition import NMF
from scipy.ndimage import uniform_filter
import time
import os


# ── canonical name map shared by all functions ──────────────────────────
_NAME_MAP = {
    "urban": "Urban",
    "salinas": "Salinas",
    "jasperridge": "JR",
    "paviau": "PaviaU",
}


# ─────────────────────────────────────────────────────────────────────────
# Legacy full-GT NMF  (DEPRECATED for inpainting)
# ─────────────────────────────────────────────────────────────────────────

def load_dataset(name):
    """Load and normalize hyperspectral dataset.  Returns (C, H*W)."""
    name = name.lower()
    if name == "salinas":
        data = scipy.io.loadmat("HSI/data/Salinas_crop.mat")['I'].astype(float)
        data = np.clip(data, 0, None)
        for i in range(204):
            data[:, :, i] /= np.max(data[:, :, i])
    elif name == "urban":
        data = scipy.io.loadmat("HSI/data/Urban_R162.mat")['Y'].astype(float)
        for i in range(162):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(162, 307, 307).transpose(2, 1, 0)
    elif name == "jasperridge":
        data = scipy.io.loadmat("HSI/data/jasperRidge2_R198.mat")['Y'].astype(float)
        for i in range(198):
            data[i, :] /= np.max(data[i, :])
        data = data.reshape(198, 100, 100).transpose(2, 1, 0)
    elif name == "paviau":
        data = scipy.io.loadmat("HSI/data/PaviaU.mat")['paviaU'].astype(float)
        for i in range(103):
            data[:, :, i] /= np.max(data[:, :, i])
        data = data[-340:, :, :]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    data = np.transpose(data, (2, 0, 1)).reshape(data.shape[2], -1)
    return data

def nmf_initialization(I, rank):
    if hasattr(I, "detach"):
        I = I.detach().float().cpu()
        if I.dim() == 4 and I.shape[0] == 1:
            I = I.squeeze(0)
        if I.dim() == 3:
            if I.shape[0] <= I.shape[-1] and I.shape[0] <= I.shape[1]:
                I = I.reshape(I.shape[0], -1)
            else:
                I = I.permute(2, 0, 1).reshape(I.shape[-1], -1)
        I = I.numpy()
    else:
        I = np.asarray(I)
        if I.ndim == 4 and I.shape[0] == 1:
            I = I.squeeze(0)
        if I.ndim == 3:
            if I.shape[0] <= I.shape[-1] and I.shape[0] <= I.shape[1]:
                I = I.reshape(I.shape[0], -1)
            else:
                I = np.transpose(I, (2, 0, 1)).reshape(I.shape[-1], -1)
    I = I.copy()
    nmf = NMF(rank, init='random', random_state=42, max_iter=12000)
    endmember = nmf.fit_transform(I).T
    abundance = nmf.components_.T
    return endmember.astype(np.float32), abundance.astype(np.float32) 
