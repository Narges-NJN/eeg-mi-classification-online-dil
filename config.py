from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Config:
    # -----------------------
    # DATA
    # -----------------------
    data_path: str = "~/mne_data/"
    subjects: List[int] = field(default_factory=lambda: list(range(1, 81)))

    # Domain-Incremental (DI) protocol
    di_train_runs: List[int] = field(default_factory=lambda: [4, 8])
    di_test_runs: List[int] = field(default_factory=lambda: [12])

    # LOSO protocol
    loso_runs: List[int] = field(default_factory=lambda: [4, 8, 12])

    # -----------------------
    # PREPROCESSING
    # -----------------------
    ablation_mode: str = "normal"  # "normal" or "highpass80"

    di_l_freq: float = 1.0
    di_h_freq: Optional[float] = 79.0

    loso_l_freq: float = 1.0
    loso_h_freq: Optional[float] = 79.0

    resample_sfreq: Optional[float] = None

    tmin: float = -0.5
    tmax: float = 4.1

    event_id: Dict[str, int] = field(default_factory=lambda: dict(left=2, right=3))

    # -----------------------
    # NORMALIZATION
    # -----------------------
    norm_eps: float = 1e-10

    # -----------------------
    # MODEL (EEGNet)
    # -----------------------
    drop_prob: float = 0.0
    F1: int = 8
    D: int = 1
    kernel_length: int = 16
    depthwise_kernel_length: int = 8
    pool_mode: str = "mean"
    batch_norm_momentum: float = 0.01

    # -----------------------
    # TRAINING
    # -----------------------
    # ✅ One step-size parameter used for BOTH online and offline DI
    # 1 = one subject per step. 2/5 groups subjects in steps.
    di_step_size: int = 1

    # Online DI optimizer params
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    online_epochs_per_subject: int = 1

    # Offline DI optimizer params
    offline_lr: float = 1e-4
    offline_weight_decay: float = 1e-1
    offline_epochs_per_subject: int = 15

    # LOSO training
    loso_lr: float = 1e-4
    loso_batch_size: int = 64
    loso_epochs: int = 30

    # -----------------------
    # OUTPUT / REPRO
    # -----------------------
    output_dir: str = "outputs"
    seed: int = 42