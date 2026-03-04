from typing import Dict, Any, List
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure(path: str):
    os.makedirs(path, exist_ok=True)


def _R_to_matrix(R: Dict[int, Dict[int, float]], subjects: List[int]) -> np.ndarray:
    M = np.zeros((len(subjects), len(subjects)), dtype=float)
    for i, si in enumerate(subjects):
        for j, sj in enumerate(subjects):
            M[i, j] = R[str(si)][str(sj)] if str(si) in R else np.nan
    return M


def compute_cl_metrics(Rm: np.ndarray, b: np.ndarray):
    T = Rm.shape[0]
    acc_curve = np.nanmean(Rm, axis=1)

    bwt_curve = np.full(T, np.nan)
    for t in range(T):
        if t == 0:
            continue
        diffs = []
        for i in range(t):
            diffs.append(Rm[t, i] - Rm[i, i])
        bwt_curve[t] = float(np.mean(diffs)) if diffs else np.nan

    fwt_curve = np.full(T, np.nan)
    for i in range(1, T):
        fwt_curve[i] = float(Rm[i - 1, i] - b[i])

    return acc_curve, bwt_curve, fwt_curve


def plot_domain_incremental(out: Dict[str, Any], out_dir: str):
    _ensure(out_dir)
    subjects = out["subjects"]
    Rm = _R_to_matrix(out["R"], subjects)
    b = np.array([out["b"][str(sid)] for sid in subjects], dtype=float)

    steps = np.arange(1, len(subjects) + 1)

    acc_curve, bwt_curve, fwt_curve = compute_cl_metrics(Rm, b)

    plt.figure()
    plt.plot(steps, acc_curve)
    plt.title("ACC curve")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=200)
    plt.close()


############################################################
# NEW FUNCTION
############################################################

def plot_step_size_accuracy_curves(root_dir: str, step_sizes: List[int], save_path: str):
    """
    Plot accuracy curves for different step sizes in one figure.

    root_dir example:
    outputs/run_20260304_123000

    step_sizes example:
    [1,2,5]
    """

    plt.figure(figsize=(7,5))

    for ss in step_sizes:

        folder = f"online_domain_incremental_step{ss}"
        path = os.path.join(root_dir, folder, "results.json")

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        with open(path) as f:
            out = json.load(f)

        subjects = out["subjects"]

        Rm = _R_to_matrix(out["R"], subjects)
        b = np.array([out["b"][str(sid)] for sid in subjects])

        acc_curve, _, _ = compute_cl_metrics(Rm, b)

        steps = np.arange(1, len(acc_curve) + 1)

        plt.plot(steps, acc_curve, label=f"step size = {ss}")

    plt.xlabel("Training Step (Domains Seen)")
    plt.ylabel("Accuracy")
    plt.title("Online Domain-Incremental Accuracy vs Step Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()