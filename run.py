
import argparse
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from config import Config
from experiments import online_domain_incremental, offline_domain_incremental, loso
from plots import (
    plot_domain_incremental,
    plot_loso,
    plot_step_size_comparison,
    plot_online_vs_offline_same_stepsize,
)
from utils import ensure_dir, save_json


ABLATION_CASES = {
    "baseline": {},
    "lr_5e-4": {"lr": "5e-4"},
    "lr_1e-3": {"lr": "1e-3"},
    "lr_5e-3": {"lr": "5e-3"},
    "lr_3e-2": {"lr": "3e-2"},
    "wd_1e-4": {"weight_decay": "1e-4"},
    "wd_1e-3": {"weight_decay": "1e-3"},
    "wd_1e-2": {"weight_decay": "1e-2"},
    "bs_8": {"batch_size": "8"},
    "bs_16": {"batch_size": "16"},
    "bs_32": {"batch_size": "32"},
    "drop_0.10": {"drop_prob": "0.10"},
    "drop_0.25": {"drop_prob": "0.25"},
    "drop_0.50": {"drop_prob": "0.50"},
    "F1_16": {"F1": "16"},
    "F1_32": {"F1": "32"},
    "D_2": {"D": "2"},
    "D_4": {"D": "4"},
    "D_8": {"D": "8"},
    "klen_32": {"kernel_length": "32"},
    "klen_64": {"kernel_length": "64"},
    "klen_128": {"kernel_length": "128"},
    "dwklen_16": {"depthwise_kernel_length": "16"},
    "dwklen_32": {"depthwise_kernel_length": "32"},
    "dwklen_64": {"depthwise_kernel_length": "64"},
    "pool_max": {"pool_mode": "max"},
}


def _summary_di(title: str, out: Dict[str, Any]) -> str:
    return (
        f"{title}\n"
        f"  ACC: {out.get('ACC_final', float('nan')):.4f}\n"
        f"  AVG: {out.get('AVG_ACC', float('nan')):.4f}\n"
        f"  BWT: {out.get('BWT_final', float('nan')):.4f}\n"
        f"  FWT: {out.get('FWT_final', float('nan')):.4f}\n"
    )


def _summary_loso(out: Dict[str, Any]) -> str:
    accs = np.array([fr["test_acc"] for fr in out["fold_results"]], dtype=float)
    losses = np.array([fr["test_loss"] for fr in out["fold_results"]], dtype=float)
    return (
        "LOSO SUMMARY\n"
        f"  mean acc: {accs.mean():.4f}  std: {accs.std(ddof=0):.4f}\n"
        f"  mean loss: {losses.mean():.4f}\n"
    )


def _parse_set_args(set_args: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _coerce_value(old_value: Any, new_str: str) -> Any:
    if isinstance(old_value, list):
        parts = [p.strip() for p in new_str.split(",") if p.strip()]
        if len(old_value) == 0:
            try:
                return [int(p) for p in parts]
            except ValueError:
                return parts
        elem = old_value[0]
        if isinstance(elem, int):
            return [int(p) for p in parts]
        if isinstance(elem, float):
            return [float(p) for p in parts]
        return parts

    if isinstance(old_value, bool):
        return new_str.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(old_value, int):
        return int(float(new_str))
    if isinstance(old_value, float):
        return float(new_str)
    if old_value is None:
        if new_str.lower() in ("none", "null"):
            return None
        try:
            if "." in new_str or "e" in new_str.lower():
                return float(new_str)
            return int(new_str)
        except ValueError:
            return new_str
    return new_str


def apply_overrides(cfg: Config, overrides: Dict[str, str]) -> Config:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Config has no attribute '{k}'")
        old = getattr(cfg, k)
        setattr(cfg, k, _coerce_value(old, v))
    return cfg


def make_root_out(cfg: Config, tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(cfg.output_dir, f"{tag}_{stamp}")
    ensure_dir(root)
    return root


def _run_one_di_and_save(cfg: Config, root_out: str, *, which: str) -> Dict[str, Any]:
    if which == "online":
        out = online_domain_incremental(cfg)
        folder = f"online_domain_incremental_step{cfg.di_step_size}"
    elif which == "offline":
        out = offline_domain_incremental(cfg)
        folder = f"offline_domain_incremental_step{cfg.di_step_size}"
    else:
        raise ValueError(which)

    out_dir = os.path.join(root_out, folder)
    ensure_dir(out_dir)
    save_json(out, os.path.join(out_dir, "results.json"))
    plot_domain_incremental(out, out_dir)

    print(_summary_di(f"{which.upper()} DI SUMMARY (step_size={cfg.di_step_size})", out))
    return out


def _parse_step_sizes(cfg: Config, raw: str) -> List[int]:
    """
    baseline -> all subjects in one step
    otherwise integers
    """
    items = [s.strip() for s in raw.split(",") if s.strip()]
    step_sizes: List[int] = []
    for s in items:
        if s.lower() == "baseline":
            step_sizes.append(len(cfg.subjects))
        else:
            step_sizes.append(int(s))
    return step_sizes


def _di_summary_row(out: Dict[str, Any], step_size: int, baseline_ss: int) -> Dict[str, Any]:
    ss_label = "baseline" if step_size == baseline_ss else step_size
    return {
        "step_size": ss_label,
        "Last_ACC(%)": 100.0 * float(out.get("ACC_final", float("nan"))),
        "Avg_ACC(%)": 100.0 * float(out.get("AVG_ACC", float("nan"))),
        "BWT(%)": 100.0 * float(out.get("BWT_final", float("nan"))),
        "FWT(%)": 100.0 * float(out.get("FWT_final", float("nan"))),
    }


def run_loso(cfg: Config, root_out: str) -> None:
    out = loso(cfg)
    out_dir = os.path.join(root_out, "loso")
    ensure_dir(out_dir)
    plot_loso(out, out_dir)
    save_json(out, os.path.join(out_dir, "results.json"))
    print(_summary_loso(out))


def main():
    parser = argparse.ArgumentParser(description="EEG MI framework runner")
    parser.add_argument("--exp", choices=["online", "offline", "loso", "all"], default="all")
    parser.add_argument("--tag", default="run", help="Output folder tag prefix")
    parser.add_argument("--ablation", choices=sorted(ABLATION_CASES.keys()), default=None)
    parser.add_argument("--set", nargs="*", default=[], help="Override config values: key=value ...")

    parser.add_argument(
        "--di_step_sizes",
        default=None,
        help="Comma-separated DI step sizes to sweep for ONLINE and/or OFFLINE (e.g. 'baseline,1,2,5').",
    )

    args = parser.parse_args()

    base_cfg = Config()

    # preset ablation first
    if args.ablation is not None:
        base_cfg = apply_overrides(base_cfg, ABLATION_CASES[args.ablation])
        if args.tag == "run":
            args.tag = f"ablation_{args.ablation}"

    overrides = _parse_set_args(args.set)
    base_cfg = apply_overrides(base_cfg, overrides)

    root_out = make_root_out(base_cfg, args.tag)

    # If sweeping, we run one loop and optionally run online/offline depending on --exp.
    if args.di_step_sizes is not None and args.exp in ("online", "offline", "all"):
        step_sizes = _parse_step_sizes(base_cfg, args.di_step_sizes)
        baseline_ss = len(base_cfg.subjects)

        online_rows, offline_rows = [], []

        for ss in step_sizes:
            cfg_run = Config()
            if args.ablation is not None:
                cfg_run = apply_overrides(cfg_run, ABLATION_CASES[args.ablation])
            cfg_run = apply_overrides(cfg_run, overrides)
            cfg_run.di_step_size = ss

            if args.exp in ("online", "all"):
                out_on = _run_one_di_and_save(cfg_run, root_out, which="online")
                online_rows.append(_di_summary_row(out_on, ss, baseline_ss))

            if args.exp in ("offline", "all"):
                out_off = _run_one_di_and_save(cfg_run, root_out, which="offline")
                offline_rows.append(_di_summary_row(out_off, ss, baseline_ss))

        sweep_dir = os.path.join(root_out, "di_step_sweep")
        ensure_dir(sweep_dir)

        if online_rows:
            save_json({"rows": online_rows}, os.path.join(sweep_dir, "online_table_summary.json"))
            print("ONLINE step-size summary rows:", online_rows)

        if offline_rows:
            save_json({"rows": offline_rows}, os.path.join(sweep_dir, "offline_table_summary.json"))
            print("OFFLINE step-size summary rows:", offline_rows)

        # -----------------------------
        # NEW: Generate overlay plots
        # -----------------------------
        plots_dir = os.path.join(sweep_dir, "comparison_plots")
        ensure_dir(plots_dir)

        # Overlay step sizes for online/offline
        if args.exp in ("online", "all") and online_rows:
            plot_step_size_comparison(
                root_out,
                which="online",
                step_sizes=step_sizes,
                out_dir=plots_dir,
                use_seen_curve_if_available=True,
            )

        if args.exp in ("offline", "all") and offline_rows:
            plot_step_size_comparison(
                root_out,
                which="offline",
                step_sizes=step_sizes,
                out_dir=plots_dir,
                use_seen_curve_if_available=True,
            )

        # Online vs Offline for a single chosen step size
        # Preference: step_size=5 if present, else last provided
        chosen = 5 if 5 in step_sizes else step_sizes[-1]
        if (args.exp == "all") and online_rows and offline_rows:
            plot_online_vs_offline_same_stepsize(
                root_out,
                step_size=chosen,
                out_dir=plots_dir,
                use_seen_curve_if_available=True,
            )

    else:
        if args.exp in ("online", "all"):
            _run_one_di_and_save(base_cfg, root_out, which="online")
        if args.exp in ("offline", "all"):
            _run_one_di_and_save(base_cfg, root_out, which="offline")
        if args.exp in ("loso", "all"):
            run_loso(base_cfg, root_out)

    print("Saved all outputs to:", root_out)


if __name__ == "__main__":
    main()