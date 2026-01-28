# data.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

from config import Config
from normalize import fit_norm_params_from_raw, apply_norm_params_to_raw


def load_raw_subject(
    cfg: Config,
    subject_id: int,
    runs: List[int],
    l_freq: float,
    h_freq: Optional[float],
) -> mne.io.BaseRaw:
    """Load + concat EDF runs; apply filter/resample; NO normalization here."""
    raw_files = eegbci.load_data(subject_id, runs, path=cfg.data_path)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_files]
    raw = mne.concatenate_raws(raws)

    if cfg.ablation_mode == "normal":
        raw = raw.copy().filter(l_freq, h_freq, verbose=False)
    elif cfg.ablation_mode == "highpass80":
        raw = raw.copy().filter(80.0, None, verbose=False)
    else:
        raise ValueError(f"Unknown ablation_mode: {cfg.ablation_mode}")

    if cfg.resample_sfreq is not None:
        raw.resample(cfg.resample_sfreq, verbose=False)

    return raw


def raw_to_epochs_Xy(cfg: Config, raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray, mne.Info]:
    """
    Convert normalized RAW -> epochs -> X,y.
    Uses event_id {left:2, right:3} and maps right(3)->1, left(2)->0.
    """
    events, _ = mne.events_from_annotations(raw, verbose=False)

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=cfg.event_id,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=None,
        picks=picks,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()  # (n_epochs, n_chans, n_times)
    y_codes = epochs.events[:, 2]
    y = (y_codes == cfg.event_id["right"]).astype(int)

    return X, y, epochs.info


def load_di_data_train_only_norm(cfg: Config) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """
    DI data loading EXACTLY like better code:
      - load raw_train (train runs) and raw_test (test runs) per subject
      - fit norm params on raw_train only
      - apply to raw_train and raw_test
      - epoch both
    """
    train_data: Dict[int, dict] = {}
    test_data: Dict[int, dict] = {}

    for subject in cfg.subjects:
        raw_train = load_raw_subject(cfg, subject, cfg.di_train_runs, cfg.di_l_freq, cfg.di_h_freq)
        raw_test = load_raw_subject(cfg, subject, cfg.di_test_runs, cfg.di_l_freq, cfg.di_h_freq)

        mean_ch, std_ch, min_ch, max_ch = fit_norm_params_from_raw(raw_train, eps=cfg.norm_eps)

        raw_train = apply_norm_params_to_raw(raw_train, mean_ch, std_ch, min_ch, max_ch, eps=cfg.norm_eps)
        raw_test = apply_norm_params_to_raw(raw_test, mean_ch, std_ch, min_ch, max_ch, eps=cfg.norm_eps)

        X_train, y_train, info = raw_to_epochs_Xy(cfg, raw_train)
        X_test, y_test, _ = raw_to_epochs_Xy(cfg, raw_test)

        train_data[subject] = {"X": X_train, "y": y_train, "info": info}
        test_data[subject] = {"X": X_test, "y": y_test}

    return train_data, test_data


def load_loso_subject_epochs_subjectwise_norm(cfg: Config, subject: int) -> Tuple[np.ndarray, np.ndarray, mne.Info]:
    """
    LOSO (cross-sub style): normalize per subject using that subject's own RAW.
    Then epoch. No leakage across subjects, and matches cross_sub behavior.
    """
    raw = load_raw_subject(cfg, subject, cfg.loso_runs, cfg.loso_l_freq, cfg.loso_h_freq)

    mean_ch, std_ch, min_ch, max_ch = fit_norm_params_from_raw(raw, eps=cfg.norm_eps)
    raw = apply_norm_params_to_raw(raw, mean_ch, std_ch, min_ch, max_ch, eps=cfg.norm_eps)

    X, y, info = raw_to_epochs_Xy(cfg, raw)
    return X, y, info
