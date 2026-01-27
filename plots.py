from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)

def _R_to_matrix(R: Dict[int, Dict[int, float]], subjects: List[int]) -> np.ndarray:
    M = np.zeros((len(subjects), len(subjects)), dtype=float)
    for i, si in enumerate(subjects):
        for j, sj in enumerate(subjects):
            M[i, j] = R[si].get(sj, np.nan)
    return M

def compute_cl_metrics(Rm: np.ndarray, b: np.ndarray):
    # Rm: (T,T) after each task i eval on task j
    # b: (T,) baseline before learning any task
    T = Rm.shape[0]
    acc_curve = np.nanmean(Rm, axis=1)

    # BWT at step t: mean_{i<t} (R[t,i] - R[i,i])
    bwt_curve = np.full(T, np.nan)
    for t in range(T):
        if t == 0:
            continue
        diffs = []
        for i in range(t):
            diffs.append(Rm[t, i] - Rm[i, i])
        bwt_curve[t] = float(np.mean(diffs)) if diffs else np.nan

    # FWT at step t: mean_{i>t} (R[t,i] - b[i]) doesn't exist for last steps; plot per i pre-learn
    # Simple classic: for i>=1, use R[i-1, i] - b[i]
    fwt_curve = np.full(T, np.nan)
    for i in range(1, T):
        fwt_curve[i] = float(Rm[i-1, i] - b[i])

    return acc_curve, bwt_curve, fwt_curve

def plot_domain_incremental(out: Dict[str, Any], out_dir: str):
    _ensure(out_dir)
    subjects = out["subjects"]
    Rm = _R_to_matrix(out["R"], subjects)
    b = np.array([out["b"][sid] for sid in subjects], dtype=float)

    # 1) Heatmap R
    plt.figure()
    sns.heatmap(Rm, vmin=0, vmax=1, cmap="viridis")
    plt.title("Accuracy Matrix R (after training on row-subject, test on col-subject)")
    plt.xlabel("Test subject")
    plt.ylabel("Train step subject")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "R_heatmap.png"), dpi=200)
    plt.close()

    # 2) ACC / BWT / FWT curves
    acc_curve, bwt_curve, fwt_curve = compute_cl_metrics(Rm, b)
    steps = np.arange(1, len(subjects) + 1)

    plt.figure()
    plt.plot(steps, acc_curve)
    plt.title("ACC curve (mean accuracy over all subjects vs step)")
    plt.xlabel("Step (subjects seen)")
    plt.ylabel("ACC")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(steps, bwt_curve)
    plt.title("BWT curve")
    plt.xlabel("Step")
    plt.ylabel("BWT")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bwt_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(steps, fwt_curve)
    plt.title("FWT curve (R[i-1,i] - b[i])")
    plt.xlabel("Step")
    plt.ylabel("FWT")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fwt_curve.png"), dpi=200)
    plt.close()

    # 3) Trajectories: accuracy on each subject vs step (spaghetti)
    plt.figure()
    for j, sj in enumerate(subjects):
        plt.plot(steps, Rm[:, j], alpha=0.5)
    plt.title("Per-subject test accuracy trajectory across steps")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectories.png"), dpi=200)
    plt.close()

    # 4) Forgetting heatmap: max past - current for each task j at each step t
    forget = np.zeros_like(Rm)
    for j in range(Rm.shape[1]):
        best = -np.inf
        for t in range(Rm.shape[0]):
            best = max(best, Rm[t, j])
            forget[t, j] = best - Rm[t, j]
    plt.figure()
    sns.heatmap(forget, cmap="magma")
    plt.title("Forgetting heatmap (best_past - current)")
    plt.xlabel("Test subject")
    plt.ylabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forgetting_heatmap.png"), dpi=200)
    plt.close()

    # 5) Final accuracies bar
    final_acc = Rm[-1, :]
    plt.figure()
    plt.bar(np.arange(len(subjects)), final_acc)
    plt.title("Final accuracy per subject")
    plt.xlabel("Subject index")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_acc_bar.png"), dpi=200)
    plt.close()

    # Save CSV
    pd.DataFrame(Rm, index=subjects, columns=subjects).to_csv(os.path.join(out_dir, "R.csv"))
    pd.DataFrame({"subject": subjects, "baseline_b": b}).to_csv(os.path.join(out_dir, "baseline_b.csv"), index=False)

def plot_loso(out: Dict[str, Any], out_dir: str):
    _ensure(out_dir)
    df = pd.DataFrame(out["fold_results"])
    df.to_csv(os.path.join(out_dir, "fold_results.csv"), index=False)

    # 1) Fold accuracies bar
    plt.figure()
    plt.bar(df["test_subject"].astype(int).values, df["test_acc"].values)
    plt.title("LOSO accuracy per held-out subject")
    plt.xlabel("Held-out subject")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loso_acc_bar.png"), dpi=200)
    plt.close()

    # 2) Fold loss bar
    plt.figure()
    plt.bar(df["test_subject"].astype(int).values, df["test_loss"].values)
    plt.title("LOSO test loss per held-out subject")
    plt.xlabel("Held-out subject")
    plt.ylabel("Cross-entropy loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loso_loss_bar.png"), dpi=200)
    plt.close()

    # 3) Aggregate confusion matrix
    all_y, all_pred, all_prob = [], [], []
    for sid, pack in out["fold_predictions"].items():
        all_y.append(pack["y"])
        all_pred.append(pack["pred"])
        all_prob.append(pack["prob"][:, 1])
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    prob1 = np.concatenate(all_prob)

    cm = confusion_matrix(y, pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("LOSO aggregated confusion matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loso_confusion.png"), dpi=200)
    plt.close()

    # 4) ROC curve (binary)
    fpr, tpr, _ = roc_curve(y, prob1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"LOSO ROC (AUC={roc_auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loso_roc.png"), dpi=200)
    plt.close()

    # 5) Probability histogram (confidence)
    plt.figure()
    plt.hist(prob1, bins=30)
    plt.title("LOSO predicted P(class=1) histogram")
    plt.xlabel("P(right)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loso_prob_hist.png"), dpi=200)
    plt.close()
