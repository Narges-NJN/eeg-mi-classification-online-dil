# experiments.py
from __future__ import annotations

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config
from data import load_di_data_train_only_norm, load_loso_subject_epochs_subjectwise_norm
from models import build_model
from train_utils import train_one_epoch, evaluate_full


# ---------------------------
# Dataset
# ---------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# Utilities
# ---------------------------
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _compute_baseline_b(
    cfg: Config,
    model_init_state: Dict[str, torch.Tensor],
    n_chans: int,
    n_times: int,
    test_data: Dict[int, Dict[str, Any]],
    device: str,
) -> Dict[int, float]:
    """
    Compute b[j] = accuracy of randomly initialized model on subject j test set.
    Matches your online DI baseline logic.
    """
    criterion = nn.CrossEntropyLoss()
    model_b = build_model(cfg, n_chans=n_chans, n_times=n_times, n_classes=2).to(device)
    model_b.load_state_dict(model_init_state)

    b: Dict[int, float] = {}
    for subj in cfg.subjects:
        ds = EEGDataset(test_data[subj]["X"], test_data[subj]["y"])
        ld = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        ev = evaluate_full(model_b, ld, criterion, device)
        b[subj] = float(ev["acc"])
    return b


def _compute_final_metrics(
    subjects: List[int],
    rows: List[Dict[int, float]],
    b: Dict[int, float],
) -> Tuple[float, float, float]:
    """
    Compute ACC/BWT/FWT using the same definitions as your online DI code.
    """
    T = len(subjects)
    R_full = np.zeros((T, T), dtype=float)
    for i in range(T):
        for j, subj in enumerate(subjects):
            R_full[i, j] = rows[i][subj]

    b_vec = np.array([b[s] for s in subjects], dtype=float)

    ACC_final = float(np.mean(R_full[T - 1, :]))
    BWT_final = float(np.mean([R_full[T - 1, i] - R_full[i, i] for i in range(T - 1)])) if T > 1 else 0.0
    FWT_final = float(np.mean([R_full[i - 1, i] - b_vec[i] for i in range(1, T)])) if T > 1 else 0.0
    return ACC_final, BWT_final, FWT_final


# ---------------------------
# Shared pure DIL engine
# ---------------------------
def _run_pure_domain_incremental(
    cfg: Config,
    *,
    epochs_per_subject: int,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    desc: str = "DI",
) -> Dict[str, Any]:
    """
    PURE domain-incremental learning:
      - sequential subjects/domains
      - train ONLY on the current subject (no replay, no concatenation)
      - evaluate on all subjects at each step
    The only intended difference between "online" and "offline" is epochs_per_subject.
    """
    set_global_seed(cfg.seed)
    device = _device()

    train_data, test_data = load_di_data_train_only_norm(cfg)

    sample_X = train_data[cfg.subjects[0]]["X"]
    n_chans, n_times = sample_X.shape[1], sample_X.shape[2]

    model = build_model(cfg, n_chans=n_chans, n_times=n_times, n_classes=2).to(device)
    init_state = copy.deepcopy(model.state_dict())

    # Baseline b from random init
    b = _compute_baseline_b(cfg, init_state, n_chans, n_times, test_data, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=(cfg.lr if lr is None else lr),
        weight_decay=(cfg.weight_decay if weight_decay is None else weight_decay),
    )

    R: Dict[int, Dict[int, float]] = {}
    history: List[Dict[str, Any]] = []
    rows: List[Dict[int, float]] = []

    subjects = cfg.subjects

    for step, train_subject in enumerate(tqdm(subjects, desc=desc, total=len(subjects))):
        Xtr = train_data[train_subject]["X"]
        ytr = train_data[train_subject]["y"]

        train_ds = EEGDataset(Xtr, ytr)
        train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        # Train on CURRENT subject only (PURE)
        for _ in range(epochs_per_subject):
            train_loss, train_acc = train_one_epoch(model, train_ld, criterion, optimizer, device)

        # Evaluate on all subjects
        row: Dict[int, float] = {}
        for test_subject in subjects:
            Xte = test_data[test_subject]["X"]
            yte = test_data[test_subject]["y"]
            test_ds = EEGDataset(Xte, yte)
            test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
            ev = evaluate_full(model, test_ld, criterion, device)
            row[test_subject] = float(ev["acc"])

        rows.append(row)
        R[train_subject] = {s: float(row[s]) for s in subjects}

        # Prefix metrics (same as before)
        T = len(subjects)
        Rm = np.full((step + 1, T), np.nan, dtype=float)
        for i in range(step + 1):
            for j, subj in enumerate(subjects):
                Rm[i, j] = rows[i][subj]
        b_vec = np.array([b[s] for s in subjects], dtype=float)

        ACC_t = float(np.nanmean(Rm[step, :]))
        if step > 0:
            BWT_t = float(np.mean([Rm[step, i] - Rm[i, i] for i in range(step)]))
            FWT_t = float(np.mean([Rm[i - 1, i] - b_vec[i] for i in range(1, step + 1)]))
        else:
            BWT_t, FWT_t = 0.0, 0.0

        history.append({
            "train_subject": int(train_subject),
            "loss": float(train_loss),
            "acc": float(train_acc),
            "row_mean_acc": float(ACC_t),
            "BWT_t": float(BWT_t),
            "FWT_t": float(FWT_t),
            "epochs_per_subject": int(epochs_per_subject),
            "train_size": int(Xtr.shape[0]),
        })

    ACC_final, BWT_final, FWT_final = _compute_final_metrics(subjects, rows, b)

    return {
        "subjects": subjects,
        "b": b,
        "R": R,
        "history": history,
        "ACC_final": ACC_final,
        "BWT_final": BWT_final,
        "FWT_final": FWT_final,
    }


# ---------------------------
# Public experiments
# ---------------------------
def online_domain_incremental(cfg: Config) -> Dict[str, Any]:
    out = _run_pure_domain_incremental(
        cfg,
        epochs_per_subject=cfg.online_epochs_per_subject,
        desc="Online DI",
    )
    out["experiment"] = "online_domain_incremental"
    return out


def offline_domain_incremental(cfg: Config) -> Dict[str, Any]:
    """
    PURE offline DIL: identical to online except epochs_per_subject (offline = more epochs).
    """
    offline_lr = getattr(cfg, "offline_lr", cfg.offline_lr)
    offline_wd = getattr(cfg, "offline_weight_decay", cfg.offline_weight_decay)

    out = _run_pure_domain_incremental(
        cfg,
        epochs_per_subject=cfg.offline_epochs_per_subject,
        lr=offline_lr,
        weight_decay=offline_wd,
        desc="Offline DI (pure)",
    )
    out["experiment"] = "offline_domain_incremental"
    return out


def loso(cfg: Config) -> Dict[str, Any]:
    """
    LOSO: per-subject RAW normalization before epoching + EEGNet + PyTorch training.

    Output format matches plots.py:
      out["fold_results"] list of dict with test_subject, test_acc, test_loss
      out["fold_predictions"] dict[str(sid)] -> {y, pred, prob}
    """
    set_global_seed(cfg.seed)
    device = _device()

    # LOSO-specific hyperparams if present
    loso_lr = getattr(cfg, "loso_lr", cfg.lr)
    loso_wd = getattr(cfg, "loso_weight_decay", cfg.weight_decay)
    loso_bs = getattr(cfg, "loso_batch_size", cfg.batch_size)
    loso_epochs = getattr(cfg, "loso_epochs", cfg.loso_epochs)
    loso_num_workers = getattr(cfg, "loso_num_workers", 0)
    pin_memory = (device == "cuda")

    # Precompute subject-wise normalized epochs once
    X_by_subj: Dict[int, np.ndarray] = {}
    y_by_subj: Dict[int, np.ndarray] = {}
    for s in cfg.subjects:
        Xs, ys, _ = load_loso_subject_epochs_subjectwise_norm(cfg, s)
        X_by_subj[s] = Xs
        y_by_subj[s] = ys

    any_s = cfg.subjects[0]
    n_chans, n_times = X_by_subj[any_s].shape[1], X_by_subj[any_s].shape[2]

    fold_results: List[Dict[str, Any]] = []
    fold_predictions: Dict[str, Dict[str, Any]] = {}

    for test_subject in tqdm(cfg.subjects, desc="LOSO folds", total=len(cfg.subjects)):
        Xtr = np.concatenate([X_by_subj[s] for s in cfg.subjects if s != test_subject], axis=0)
        ytr = np.concatenate([y_by_subj[s] for s in cfg.subjects if s != test_subject], axis=0)
        Xte = X_by_subj[test_subject]
        yte = y_by_subj[test_subject]

        model = build_model(cfg, n_chans=n_chans, n_times=n_times, n_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=loso_lr, weight_decay=loso_wd)

        train_ds = EEGDataset(Xtr, ytr)
        train_ld = DataLoader(
            train_ds, batch_size=loso_bs, shuffle=True,
            num_workers=loso_num_workers, pin_memory=pin_memory
        )
        test_ds = EEGDataset(Xte, yte)
        test_ld = DataLoader(
            test_ds, batch_size=loso_bs, shuffle=False,
            num_workers=loso_num_workers, pin_memory=pin_memory
        )

        for _ in tqdm(range(loso_epochs), desc=f"  Train fold {test_subject}", leave=False):
            train_loss, train_acc = train_one_epoch(model, train_ld, criterion, optimizer, device)

        ev = evaluate_full(model, test_ld, criterion, device)
        fold_results.append({
            "test_subject": int(test_subject),
            "test_acc": float(ev["acc"]),
            "test_loss": float(ev["loss"]),
        })

        # Keep numpy arrays in memory; run.py's JSON saver can serialize them
        fold_predictions[str(test_subject)] = {
            "y": ev["y"].astype(int),
            "pred": ev["pred"].astype(int),
            "prob": ev["prob"].astype(float),
        }

    return {
        "experiment": "loso",
        "subjects": cfg.subjects,
        "fold_results": fold_results,
        "fold_predictions": fold_predictions,
    }
