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


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def _make_task_groups(subjects: List[int], step_size: int) -> List[List[int]]:
    if step_size <= 0:
        raise ValueError(f"step_size must be >= 1, got {step_size}")
    return [subjects[i:i + step_size] for i in range(0, len(subjects), step_size)]


def _concat_train_data_for_subjects(train_data: Dict[int, Dict[str, Any]], group: List[int]):
    Xs, ys = [], []
    for s in group:
        Xs.append(train_data[s]["X"])
        ys.append(train_data[s]["y"])
    X = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
    y = np.concatenate(ys, axis=0) if len(ys) > 1 else ys[0]
    return X, y


def _compute_baseline_b_task(
    cfg: Config,
    model_init_state: Dict[str, torch.Tensor],
    n_chans: int,
    n_times: int,
    test_data: Dict[int, Dict[str, Any]],
    task_groups: List[List[int]],
    device: str,
) -> Dict[int, float]:
    criterion = nn.CrossEntropyLoss()
    model_b = build_model(cfg, n_chans=n_chans, n_times=n_times, n_classes=2).to(device)
    model_b.load_state_dict(model_init_state)

    b_task: Dict[int, float] = {}
    for t, group in enumerate(task_groups):
        accs = []
        for subj in group:
            ds = EEGDataset(test_data[subj]["X"], test_data[subj]["y"])
            ld = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
            ev = evaluate_full(model_b, ld, criterion, device)
            accs.append(float(ev["acc"]))
        b_task[t] = float(np.mean(accs)) if accs else 0.0
    return b_task


def _rows_subject_to_task(
    rows_subject: List[Dict[int, float]],
    task_groups: List[List[int]],
) -> List[Dict[int, float]]:
    rows_task: List[Dict[int, float]] = []
    for row in rows_subject:
        rt: Dict[int, float] = {}
        for t, group in enumerate(task_groups):
            rt[t] = float(np.mean([row[s] for s in group]))
        rows_task.append(rt)
    return rows_task


def _compute_final_metrics_task(
    n_tasks: int,
    rows_task: List[Dict[int, float]],
    b_task: Dict[int, float],
) -> Tuple[float, float, float]:
    """
    --- FIX ---
    Compute metrics robustly even if we don't have a perfect T x T (e.g., interrupted runs).
    Previously you forced BWT/FWT to 0 unless len(rows_task) == T.
    """
    T = n_tasks
    if len(rows_task) == 0:
        return float("nan"), 0.0, 0.0

    R_full = np.full((len(rows_task), T), np.nan, dtype=float)
    for i in range(len(rows_task)):
        for t in range(T):
            # rows_task[i] should contain every task key; use .get for safety
            R_full[i, t] = float(rows_task[i].get(t, np.nan))

    b_vec = np.array([float(b_task.get(t, np.nan)) for t in range(T)], dtype=float)

    last = len(rows_task) - 1
    ACC_final = float(np.nanmean(R_full[last, :]))

    if T <= 1:
        return ACC_final, 0.0, 0.0

    # BWT needs diagonal R[i,i] and last row R[last,i]
    max_i_for_bwt = min(T - 1, last)  # i in [0..T-2], but also must have row i
    if max_i_for_bwt >= 0:
        diffs = []
        for i in range(max_i_for_bwt + 1):
            if i >= T - 1:
                break
            a = R_full[last, i]
            d = R_full[i, i]
            if np.isfinite(a) and np.isfinite(d):
                diffs.append(a - d)
        BWT_final = float(np.mean(diffs)) if diffs else 0.0
    else:
        BWT_final = 0.0

    # FWT uses R[i-1,i] - b[i] for i=1..T-1, but we need rows up to i-1
    max_i_for_fwt = min(T - 1, last)  # i requires row (i-1) so last>=i-1
    diffs = []
    for i in range(1, max_i_for_fwt + 1):
        prev = R_full[i - 1, i]
        base = b_vec[i]
        if np.isfinite(prev) and np.isfinite(base):
            diffs.append(prev - base)
    FWT_final = float(np.mean(diffs)) if diffs else 0.0

    return ACC_final, BWT_final, FWT_final


def _run_pure_domain_incremental(
    cfg: Config,
    *,
    epochs_per_subject: int,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    step_size: int = 1,
    desc: str = "DI",
) -> Dict[str, Any]:
    set_global_seed(cfg.seed)
    device = _device()

    train_data, test_data = load_di_data_train_only_norm(cfg)

    sample_X = train_data[cfg.subjects[0]]["X"]
    n_chans, n_times = sample_X.shape[1], sample_X.shape[2]

    model = build_model(cfg, n_chans=n_chans, n_times=n_times, n_classes=2).to(device)
    init_state = copy.deepcopy(model.state_dict())

    subjects = cfg.subjects
    task_groups = _make_task_groups(subjects, step_size)
    n_tasks = len(task_groups)

    b_task = _compute_baseline_b_task(cfg, init_state, n_chans, n_times, test_data, task_groups, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=(cfg.lr if lr is None else lr),
        weight_decay=(cfg.weight_decay if weight_decay is None else weight_decay),
    )

    rows_subject: List[Dict[int, float]] = []
    history: List[Dict[str, Any]] = []
    R_task: Dict[int, Dict[int, float]] = {}
    acc_seen_tasks: List[float] = []

    for t, group in enumerate(tqdm(task_groups, desc=desc, total=n_tasks)):
        Xtr, ytr = _concat_train_data_for_subjects(train_data, group)
        train_ds = EEGDataset(Xtr, ytr)
        train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs_per_subject):
            train_loss, train_acc = train_one_epoch(model, train_ld, criterion, optimizer, device)

        row_subj: Dict[int, float] = {}
        for test_subject in subjects:
            Xte = test_data[test_subject]["X"]
            yte = test_data[test_subject]["y"]
            test_ds = EEGDataset(Xte, yte)
            test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
            ev = evaluate_full(model, test_ld, criterion, device)
            row_subj[test_subject] = float(ev["acc"])
        rows_subject.append(row_subj)

        row_task = {
            task_i: float(np.mean([row_subj[s] for s in task_groups[task_i]]))
            for task_i in range(n_tasks)
        }
        R_task[t] = row_task

        acc_t_seen = float(np.mean([row_task[i] for i in range(t + 1)]))
        acc_seen_tasks.append(acc_t_seen)

        if t > 0:
            bwt_t = float(np.mean([R_task[t][i] - R_task[i][i] for i in range(t)]))
            fwt_t = float(np.mean([R_task[i - 1][i] - b_task[i] for i in range(1, t + 1)]))
        else:
            bwt_t, fwt_t = 0.0, 0.0

        history.append({
            "train_task": int(t),
            "train_subjects": [int(s) for s in group],
            "loss": float(train_loss),
            "acc": float(train_acc),
            "row_mean_acc_seen_tasks": float(acc_t_seen),
            "BWT_t": float(bwt_t),
            "FWT_t": float(fwt_t),
            "epochs_per_subject": int(epochs_per_subject),
            "train_size": int(Xtr.shape[0]),
            "step_size": int(step_size),
        })

    rows_task = _rows_subject_to_task(rows_subject, task_groups)
    ACC_final, BWT_final, FWT_final = _compute_final_metrics_task(n_tasks, rows_task, b_task)
    AVG_ACC = float(np.mean(acc_seen_tasks)) if acc_seen_tasks else float("nan")

    return {
        "subjects": list(range(n_tasks)),          # task indices (not raw subject ids)
        "task_groups": task_groups,               # mapping of tasks -> subject ids
        "b": b_task,                              # baseline per task
        "R": R_task,                              # task->task accuracy dict
        "history": history,
        "step_size": int(step_size),
        "ACC_final": ACC_final,
        "AVG_ACC": AVG_ACC,
        "BWT_final": BWT_final,
        "FWT_final": FWT_final,
    }


def online_domain_incremental(cfg: Config) -> Dict[str, Any]:
    out = _run_pure_domain_incremental(
        cfg,
        epochs_per_subject=cfg.online_epochs_per_subject,
        step_size=getattr(cfg, "di_step_size", 1),
        desc="Online DI",
    )
    out["experiment"] = "online_domain_incremental"
    return out


def offline_domain_incremental(cfg: Config) -> Dict[str, Any]:
    out = _run_pure_domain_incremental(
        cfg,
        epochs_per_subject=cfg.offline_epochs_per_subject,
        lr=cfg.offline_lr,
        weight_decay=cfg.offline_weight_decay,
        step_size=getattr(cfg, "di_step_size", 1),
        desc="Offline DI (pure)",
    )
    out["experiment"] = "offline_domain_incremental"
    return out


def loso(cfg: Config) -> Dict[str, Any]:
    """
    --- FIX ---
    Your LOSO was broken because:
      - data.load_loso_subject_epochs_subjectwise_norm expects (cfg, subject)
        but you called it without subject.
      - train_utils.evaluate_full returns keys: y, pred, prob
        but you used y_true/y_pred/y_prob.
    """
    set_global_seed(cfg.seed)
    device = _device()

    criterion = nn.CrossEntropyLoss()

    fold_results: List[Dict[str, Any]] = []
    fold_predictions: Dict[str, Dict[str, Any]] = {}

    subjects = cfg.subjects
    # Build per-subject normalized epochs
    data: Dict[int, Dict[str, Any]] = {}
    for s in subjects:
        X, y, info = load_loso_subject_epochs_subjectwise_norm(cfg, s)
        data[s] = {"X": X, "y": y, "info": info}

    for test_subject in tqdm(subjects, desc="LOSO", total=len(subjects)):
        Xtr, ytr = [], []
        for s in subjects:
            if s == test_subject:
                continue
            Xtr.append(data[s]["X"])
            ytr.append(data[s]["y"])
        Xtr = np.concatenate(Xtr, axis=0)
        ytr = np.concatenate(ytr, axis=0)

        Xte = data[test_subject]["X"]
        yte = data[test_subject]["y"]

        n_chans, n_times = Xtr.shape[1], Xtr.shape[2]
        model = build_model(cfg, n_chans=n_chans, n_times=n_times, n_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.loso_lr, weight_decay=0.0)

        train_ds = EEGDataset(Xtr, ytr)
        train_ld = DataLoader(train_ds, batch_size=cfg.loso_batch_size, shuffle=True, num_workers=0)

        test_ds = EEGDataset(Xte, yte)
        test_ld = DataLoader(test_ds, batch_size=cfg.loso_batch_size, shuffle=False, num_workers=0)

        for _ in range(cfg.loso_epochs):
            train_one_epoch(model, train_ld, criterion, optimizer, device)

        ev = evaluate_full(model, test_ld, criterion, device)

        fold_results.append({
            "test_subject": int(test_subject),
            "test_acc": float(ev["acc"]),
            "test_loss": float(ev["loss"]),
        })

        fold_predictions[str(test_subject)] = {
            "y": [int(v) for v in ev["y"]],
            "pred": [int(v) for v in ev["pred"]],
            "prob": [[float(x) for x in row] for row in ev["prob"]],
        }

    return {
        "fold_results": fold_results,
        "fold_predictions": fold_predictions,
        "subjects": subjects,
        "experiment": "loso",
    }