from typing import Dict, Any, List, Tuple
import copy
import numpy as np
import torch
from tqdm import tqdm

import data as data_mod
import normalize as norm_mod
from models import make_eegnet
from train_utils import make_loader, train_one_epoch, evaluate

def _init_model_and_baseline(cfg, device, n_chans, n_times, n_classes, test_sets: Dict[int, Tuple[np.ndarray, np.ndarray]]):
    model = make_eegnet(n_chans, n_times, n_classes, drop_prob=0.0).to(device)
    init_state = copy.deepcopy(model.state_dict())

    # baseline b: evaluate random init on each subject test set
    b = {}
    for sid, (Xte, yte) in test_sets.items():
        te_loader = make_loader(Xte, yte, cfg.batch_size, shuffle=False)
        res = evaluate(model, te_loader, device)
        b[sid] = res["acc"]

    return model, init_state, b

def run_online_domain_incremental(cfg, device) -> Dict[str, Any]:
    # Preload normalized train/test per subject using TRAIN-ONLY normalization
    train_sets = {}
    test_sets = {}
    info0 = None

    for sid in tqdm(cfg.subjects, desc="Loading DI subjects"):
        pack = data_mod.load_subject_train_test(
            sid, cfg.train_runs, cfg.test_runs,
            cfg.l_freq, cfg.h_freq, cfg.tmin, cfg.tmax,
            resample_sfreq=cfg.resample_sfreq, picks=cfg.picks
        )
        Xtr, ytr, Xte, yte, info, norm_params = norm_mod.normalize_train_test_from_raw(
            pack["raw_train"], pack["raw_test"],
            cfg.tmin, cfg.tmax, cfg.picks,
            cfg.do_zscore, cfg.do_minmax, cfg.eps, cfg.clip_z
        )
        if info0 is None:
            info0 = info
        train_sets[sid] = (Xtr, ytr)
        test_sets[sid] = (Xte, yte)

    n_chans = train_sets[cfg.subjects[0]][0].shape[1]
    n_times = train_sets[cfg.subjects[0]][0].shape[2]
    n_classes = 2

    model, init_state, b = _init_model_and_baseline(cfg, device, n_chans, n_times, n_classes, test_sets)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # R[i, j] stored as dict-of-dicts for easy subject indexing
    R = {sid: {} for sid in cfg.subjects}
    history = []

    for sid_train in tqdm(cfg.subjects, desc="Online DI"):
        Xtr, ytr = train_sets[sid_train]
        tr_loader = make_loader(Xtr, ytr, cfg.batch_size, shuffle=True)

        # online = 1 epoch
        tr_res = train_one_epoch(model, tr_loader, opt, device, cfg.grad_clip)

        # evaluate on all test subjects
        row = {}
        for sid_test in cfg.subjects:
            Xte, yte = test_sets[sid_test]
            te_loader = make_loader(Xte, yte, cfg.batch_size, shuffle=False)
            ev = evaluate(model, te_loader, device)
            row[sid_test] = ev["acc"]
            R[sid_train][sid_test] = ev["acc"]

        history.append({"train_subject": sid_train, **tr_res, "row_mean_acc": float(np.mean(list(row.values()))) })

    return {"experiment": "online_domain_incremental", "R": R, "b": b, "history": history, "subjects": cfg.subjects}

def run_offline_domain_incremental(cfg, device) -> Dict[str, Any]:
    train_sets = {}
    test_sets = {}
    info0 = None

    for sid in tqdm(cfg.subjects, desc="Loading DI subjects"):
        pack = data_mod.load_subject_train_test(
            sid, cfg.train_runs, cfg.test_runs,
            cfg.l_freq, cfg.h_freq, cfg.tmin, cfg.tmax,
            resample_sfreq=cfg.resample_sfreq, picks=cfg.picks
        )
        Xtr, ytr, Xte, yte, info, _ = norm_mod.normalize_train_test_from_raw(
            pack["raw_train"], pack["raw_test"],
            cfg.tmin, cfg.tmax, cfg.picks,
            cfg.do_zscore, cfg.do_minmax, cfg.eps, cfg.clip_z
        )
        if info0 is None:
            info0 = info
        train_sets[sid] = (Xtr, ytr)
        test_sets[sid] = (Xte, yte)

    n_chans = train_sets[cfg.subjects[0]][0].shape[1]
    n_times = train_sets[cfg.subjects[0]][0].shape[2]
    n_classes = 2

    model, init_state, b = _init_model_and_baseline(cfg, device, n_chans, n_times, n_classes, test_sets)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    R = {sid: {} for sid in cfg.subjects}
    history = []

    for sid_train in tqdm(cfg.subjects, desc="Offline DI"):
        Xtr, ytr = train_sets[sid_train]
        tr_loader = make_loader(Xtr, ytr, cfg.batch_size, shuffle=True)

        epoch_logs = []
        for ep in range(cfg.offline_epochs_per_subject):
            tr_res = train_one_epoch(model, tr_loader, opt, device, cfg.grad_clip)
            epoch_logs.append({"epoch": ep + 1, **tr_res})

        row = {}
        for sid_test in cfg.subjects:
            Xte, yte = test_sets[sid_test]
            te_loader = make_loader(Xte, yte, cfg.batch_size, shuffle=False)
            ev = evaluate(model, te_loader, device)
            row[sid_test] = ev["acc"]
            R[sid_train][sid_test] = ev["acc"]

        history.append({
            "train_subject": sid_train,
            "final_epoch_loss": epoch_logs[-1]["loss"],
            "final_epoch_acc": epoch_logs[-1]["acc"],
            "epochs": epoch_logs,
            "row_mean_acc": float(np.mean(list(row.values())))
        })

    return {"experiment": "offline_domain_incremental", "R": R, "b": b, "history": history, "subjects": cfg.subjects}

def run_loso(cfg, device) -> Dict[str, Any]:
    # Preload each subject all-runs (already filtered+epoched)
    subj_data = {}
    for sid in tqdm(cfg.subjects, desc="Loading LOSO subjects"):
        pack = data_mod.load_subject_all_runs(
            sid, cfg.loso_runs,
            cfg.l_freq, cfg.h_freq, cfg.tmin, cfg.tmax,
            resample_sfreq=cfg.resample_sfreq, picks=cfg.picks
        )
        subj_data[sid] = pack

    # Infer sizes
    any_sid = cfg.subjects[0]
    X0 = subj_data[any_sid]["X"]
    n_chans, n_times = X0.shape[1], X0.shape[2]
    n_classes = 2

    # Fixed init weights for every fold (fair)
    base_model = make_eegnet(n_chans, n_times, n_classes, drop_prob=0.0).to(device)
    init_state = copy.deepcopy(base_model.state_dict())

    fold_results = []
    fold_predictions = {}

    for sid_test in tqdm(cfg.subjects, desc="LOSO folds"):
        # Build pooled train raw epochs arrays
        X_train_list, y_train_list = [], []
        for sid in cfg.subjects:
            if sid == sid_test:
                continue
            X_train_list.append(subj_data[sid]["X"])
            y_train_list.append(subj_data[sid]["y"])
        Xtr = np.concatenate(X_train_list, axis=0)
        ytr = np.concatenate(y_train_list, axis=0)

        Xte = subj_data[sid_test]["X"]
        yte = subj_data[sid_test]["y"]

        # TRAIN-ONLY normalization for LOSO:
        # Fit from Xtr, apply to Xtr and Xte (same logic as normalize.py, but we reuse apply fn)
        # We'll simulate "fit on train" stats in trial-space.
        # (This is equivalent to train-only normalization, just without raw objects.)
        from normalize import _compute_channel_stats, _compute_channel_minmax, apply_normalizer_to_xy

        params = {"do_zscore": cfg.do_zscore, "do_minmax": cfg.do_minmax, "eps": cfg.eps, "clip_z": cfg.clip_z}
        x_tmp = Xtr.astype(np.float32)

        if cfg.do_zscore:
            stats = _compute_channel_stats(x_tmp, cfg.eps)
            params.update(stats)
            xz = (x_tmp - stats["mean"]) / stats["std"]
            if cfg.clip_z is not None:
                xz = np.clip(xz, -cfg.clip_z, cfg.clip_z)
        else:
            xz = x_tmp

        if cfg.do_minmax:
            mm = _compute_channel_minmax(xz)
            params.update(mm)

        Xtr_n = apply_normalizer_to_xy(Xtr, params)
        Xte_n = apply_normalizer_to_xy(Xte, params)

        tr_loader = make_loader(Xtr_n, ytr, cfg.batch_size, shuffle=True)
        te_loader = make_loader(Xte_n, yte, cfg.batch_size, shuffle=False)

        model = make_eegnet(n_chans, n_times, n_classes, drop_prob=0.0).to(device)
        model.load_state_dict(init_state)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Train for cfg.loso_epochs
        train_curve = []
        for ep in range(cfg.loso_epochs):
            tr_res = train_one_epoch(model, tr_loader, opt, device, cfg.grad_clip)
            train_curve.append({"epoch": ep + 1, **tr_res})

        ev = evaluate(model, te_loader, device)

        fold_results.append({
            "test_subject": sid_test,
            "test_acc": ev["acc"],
            "test_loss": ev["loss"],
            "train_last_loss": train_curve[-1]["loss"],
            "train_last_acc": train_curve[-1]["acc"],
        })
        fold_predictions[sid_test] = {"y": ev["y"], "pred": ev["pred"], "prob": ev["prob"], "train_curve": train_curve}

    return {"experiment": "loso", "fold_results": fold_results, "fold_predictions": fold_predictions, "subjects": cfg.subjects}
