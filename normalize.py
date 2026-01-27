from typing import Dict, Any, Optional, Tuple
import numpy as np
import mne

def _compute_channel_stats(x: np.ndarray, eps: float) -> Dict[str, np.ndarray]:
    # x: (trials, chans, times)
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True) + eps
    return {"mean": mean, "std": std}

def _compute_channel_minmax(x: np.ndarray) -> Dict[str, np.ndarray]:
    # after zscore typically
    x_min = x.min(axis=(0, 2), keepdims=True)
    x_max = x.max(axis=(0, 2), keepdims=True)
    return {"min": x_min, "max": x_max}

def fit_normalizer_from_raw(raw_train: mne.io.BaseRaw, tmin: float, tmax: float, picks: str,
                            do_zscore: bool, do_minmax: bool, eps: float,
                            clip_z: Optional[float]) -> Dict[str, Any]:
    # Convert to epochs (train only) to estimate stats at trial-level scale
    X_train, _, _ = _raw_to_xy(raw_train, tmin, tmax, picks)

    params: Dict[str, Any] = {"do_zscore": do_zscore, "do_minmax": do_minmax, "eps": eps, "clip_z": clip_z}
    x = X_train

    if do_zscore:
        stats = _compute_channel_stats(x, eps)
        params.update(stats)
        x = (x - stats["mean"]) / stats["std"]
        if clip_z is not None:
            x = np.clip(x, -clip_z, clip_z)

    if do_minmax:
        mm = _compute_channel_minmax(x)
        params.update(mm)

    return params

def apply_normalizer_to_xy(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    x = X.astype(np.float32)

    if params["do_zscore"]:
        x = (x - params["mean"]) / params["std"]
        if params["clip_z"] is not None:
            x = np.clip(x, -params["clip_z"], params["clip_z"])

    if params["do_minmax"]:
        denom = (params["max"] - params["min"])
        denom = np.where(denom == 0, 1.0, denom)
        x = 2.0 * (x - params["min"]) / denom - 1.0  # scale to [-1, 1]

    return x.astype(np.float32)

def _raw_to_xy(raw: mne.io.BaseRaw, tmin: float, tmax: float, picks: str) -> Tuple[np.ndarray, np.ndarray, mne.Info]:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # Support both 'left'/'right' and 'T1'/'T2' style labels from EEGBCI annotations
    if "left" in event_id and "right" in event_id:
        left_key, right_key = "left", "right"
    elif "T1" in event_id and "T2" in event_id:
        left_key, right_key = "T1", "T2"
    else:
        raise ValueError(f"Missing left/right (or T1/T2) in event_id keys: {list(event_id.keys())}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left": event_id[left_key], "right": event_id[right_key]},
        tmin=tmin, tmax=tmax, baseline=None, preload=True, picks=picks, verbose=False
    )
    X = epochs.get_data().astype(np.float32)
    y_event = epochs.events[:, -1]
    # Map left->0, right->1 using the resolved right_key
    y = (y_event == event_id[right_key]).astype(np.int64)
    return X, y, epochs.info

def normalize_train_test_from_raw(raw_train: mne.io.BaseRaw, raw_test: mne.io.BaseRaw,
                                  tmin: float, tmax: float, picks: str,
                                  do_zscore: bool, do_minmax: bool, eps: float,
                                  clip_z: Optional[float]):
    Xtr, ytr, info = _raw_to_xy(raw_train, tmin, tmax, picks)
    Xte, yte, _ = _raw_to_xy(raw_test, tmin, tmax, picks)

    params = fit_normalizer_from_raw(raw_train, tmin, tmax, picks, do_zscore, do_minmax, eps, clip_z)
    Xtr_n = apply_normalizer_to_xy(Xtr, params)
    Xte_n = apply_normalizer_to_xy(Xte, params)
    return Xtr_n, ytr, Xte_n, yte, info, params
