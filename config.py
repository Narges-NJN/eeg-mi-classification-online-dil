from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    # Dataset / protocol
    subjects: List[int] = None
    train_runs: List[int] = None          # used by domain-incremental
    test_runs: List[int] = None           # used by domain-incremental
    loso_runs: List[int] = None           # used by LOSO

    # Epoching
    tmin: float = -0.5
    tmax: float = 4.1

    # Preprocessing
    l_freq: float = 1.0
    # High cutoff must be strictly less than Nyquist (sfreq/2); for 160 Hz data, use < 80
    h_freq: float = 79.0                 # baseline choice: 1–79 Hz (avoids Nyquist=80 Hz)
    resample_sfreq: Optional[float] = None  # e.g. 160.0 or None
    picks: str = "eeg"                   # MNE picks

    # Normalization (TRAIN-ONLY)
    do_zscore: bool = True
    do_minmax: bool = True
    eps: float = 1e-8
    clip_z: Optional[float] = None       # e.g. 5.0 or None

    # Training
    seed: int = 42
    device: str = "cuda"                 # "cuda" or "cpu"
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip: Optional[float] = 1.0

    # Experiments
    online_epochs_per_subject: int = 1
    offline_epochs_per_subject: int = 15  
    loso_epochs: int = 30

    # Output
    out_dir: str = "outputs"
    run_name: str = "baseline"

def default_config() -> Config:
    return Config(
        subjects=list(range(1, 11)),
        train_runs=[4, 8],
        test_runs=[12],
        loso_runs=[4, 8, 12],
    )
