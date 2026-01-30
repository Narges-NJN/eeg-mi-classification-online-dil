# config.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Config:
    # -----------------------
    # DATA
    # -----------------------
    data_path: str = "~/mne_data/"
    subjects: List[int] = field(default_factory=lambda: list(range(1, 11)))

    # Domain-Incremental (DI) protocol (matches better code)
    di_train_runs: List[int] = field(default_factory=lambda: [4, 8])
    di_test_runs: List[int] = field(default_factory=lambda: [12])

    # LOSO protocol (cross-sub style: use all runs)
    loso_runs: List[int] = field(default_factory=lambda: [4, 8, 12])

    # -----------------------
    # PREPROCESSING
    # -----------------------
    ablation_mode: str = "normal"  # "normal" or "highpass80"

    # DI filter (match better code: 1–50 Hz)
    di_l_freq: float = 1.0
    di_h_freq: Optional[float] = 79.0

    # LOSO filter (cross-sub style typically uses 1–80 Hz)
    loso_l_freq: float = 1.0
    loso_h_freq: Optional[float] = 79.0

    resample_sfreq: Optional[float] = None

    # Epoch window (match better code)
    tmin: float = -0.5
    tmax: float = 4.1

    # Event mapping (match better code)
    event_id: Dict[str, int] = field(default_factory=lambda: dict(left=2, right=3))

    # -----------------------
    # NORMALIZATION
    # -----------------------
    # RAW-based normalization params
    norm_eps: float = 1e-10

    # -----------------------
    # MODEL (EEGNet as requested)
    # -----------------------
    drop_prob: float = 0.0

    # EEGNet architectural hyperparameters (for ablations)
    # Baseline values correspond to the updated ablation table (Online DI, EEGNet)
    F1: int = 8
    D: int = 1
    kernel_length: int = 16
    depthwise_kernel_length: int = 8
    pool_mode: str = "mean"  # "mean" or "max"
    batch_norm_momentum: float = 0.01

    # -----------------------
    # TRAINING (match better code for DI)
    # -----------------------
    lr: float = 1e-2
    weight_decay: float = 0.0
    batch_size: int = 4
    offline_lr: float = 1e-3
    offline_weight_decay: float = 1e-1
    online_epochs_per_subject: int = 1
    offline_epochs_per_subject: int = 10

    # LOSO training
    loso_lr: float = 1e-4              # smaller LR helps stability
    loso_batch_size: int = 64          # speed + smoother gradients
    loso_epochs: int = 30

    # -----------------------
    # OUTPUT / REPRO
    # -----------------------
    output_dir: str = "outputs"
    seed: int = 42
