# normalize.py
from typing import Tuple
import numpy as np
import mne


def fit_norm_params_from_raw(raw: mne.io.BaseRaw, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit per-channel normalization params from continuous RAW:
      z = (x - mean)/std
      scale z to [-1,1] using min/max of z
    Returns arrays shaped (n_chans, 1).
    """
    data = raw.get_data()  # (n_chans, n_samples)
    mean_ch = data.mean(axis=1, keepdims=True)
    std_ch = data.std(axis=1, keepdims=True) + eps

    data_z = (data - mean_ch) / std_ch
    min_ch = data_z.min(axis=1, keepdims=True)
    max_ch = data_z.max(axis=1, keepdims=True)

    return mean_ch, std_ch, min_ch, max_ch


def apply_norm_params_to_raw(
    raw: mne.io.BaseRaw,
    mean_ch: np.ndarray,
    std_ch: np.ndarray,
    min_ch: np.ndarray,
    max_ch: np.ndarray,
    eps: float = 1e-10,
) -> mne.io.BaseRaw:
    """Apply pre-fit RAW normalization params to RAW in-place."""
    data = raw.get_data()
    data_z = (data - mean_ch) / std_ch
    data_norm = 2.0 * (data_z - min_ch) / (max_ch - min_ch + eps) - 1.0
    raw._data[:] = data_norm
    return raw
