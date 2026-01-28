# run.py
import argparse
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from config import Config
from experiments import online_domain_incremental, offline_domain_incremental, loso
from plots import plot_domain_incremental, plot_loso
from utils import ensure_dir, save_json


# ---------------------------
# Ablation presets (Table 4.1 – Online DI, EEGNet)
# ---------------------------
ABLATION_CASES = {
    # Baseline
    "baseline": {},

    # Learning rate
    "lr_1e-3": {"lr": "1e-3"},
    "lr_3e-2": {"lr": "3e-2"},

    # Weight decay
    "wd_0": {"weight_decay": "0"},
    "wd_1e-4": {"weight_decay": "1e-4"},
    "wd_1e-2": {"weight_decay": "1e-2"},

    # Batch size
    "bs_4": {"batch_size": "4"},
    "bs_16": {"batch_size": "16"},

    # Dropout
    "drop_0.25": {"drop_prob": "0.25"},
    "drop_0.50": {"drop_prob": "0.50"},

    # Temporal filters (F1)
    "F1_4": {"F1": "4"},
    "F1_16": {"F1": "16"},

    # Spatial filters (D)
    "D_1": {"D": "1"},
    "D_4": {"D": "4"},

    # Temporal kernel length
    "klen_32": {"kernel_length": "32"},
    "klen_128": {"kernel_length": "128"},

    # Depthwise kernel length
    "dwklen_8": {"depthwise_kernel_length": "8"},
    "dwklen_32": {"depthwise_kernel_length": "32"},

    # Pooling mode
    "pool_max": {"pool_mode": "max"},

    # BatchNorm momentum
    "bnmom_0.10": {"batch_norm_momentum": "0.10"},
}


def _summary_di(title: str, out: Dict[str, Any]) -> str:
    return (
        f"{title}\n"
        f"  ACC: {out.get('ACC_final', float('nan')):.4f}\n"
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
    """
    Parse --set key=value pairs.
    Example: --set loso_lr=1e-4 --set subjects=1,2,3,4
    """
    out: Dict[str, str] = {}
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _coerce_value(old_value: Any, new_str: str) -> Any:
    """
    Best-effort coercion based on the current type in Config.
    Keeps this simple but extensible for later ablations.
    """
    # list[int] like subjects=1,2,3
    if isinstance(old_value, list):
        if len(old_value) == 0:
            # fallback: parse as ints if possible
            parts = [p.strip() for p in new_str.split(",") if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return parts
        # infer element type from first element
        elem = old_value[0]
        parts = [p.strip() for p in new_str.split(",") if p.strip()]
        if isinstance(elem, int):
            return [int(p) for p in parts]
        if isinstance(elem, float):
            return [float(p) for p in parts]
        return parts

    # Optional[float]/float/int/bool/str
    if isinstance(old_value, bool):
        return new_str.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(old_value, int):
        return int(float(new_str))  # allows "1e2" -> 100
    if isinstance(old_value, float):
        return float(new_str)
    if old_value is None:
        # try float/int/bool/None
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


def run_online(cfg: Config, root_out: str) -> None:
    out = online_domain_incremental(cfg)
    out_dir = os.path.join(root_out, "online_domain_incremental")
    ensure_dir(out_dir)
    save_json(out, os.path.join(out_dir, "results.json"))
    plot_domain_incremental(out, out_dir)
    print(_summary_di("ONLINE DI SUMMARY", out))


def run_offline(cfg: Config, root_out: str) -> None:
    out = offline_domain_incremental(cfg)
    out_dir = os.path.join(root_out, "offline_domain_incremental")
    ensure_dir(out_dir)
    save_json(out, os.path.join(out_dir, "results.json"))
    plot_domain_incremental(out, out_dir)
    print(_summary_di("OFFLINE DI SUMMARY", out))


def run_loso(cfg: Config, root_out: str) -> None:
    out = loso(cfg)
    out_dir = os.path.join(root_out, "loso")
    ensure_dir(out_dir)
    plot_loso(out, out_dir)
    save_json(out, os.path.join(out_dir, "results.json"))
    print(_summary_loso(out))


def main():
    parser = argparse.ArgumentParser(description="EEG MI framework runner")
    parser.add_argument(
        "--exp",
        choices=["online", "offline", "loso", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--tag",
        default="run",
        help="Output folder tag prefix (e.g., ablation name)",
    )

    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATION_CASES.keys()),
        default=None,
        help=(
            "Ablation preset (Table 4.1). Applied BEFORE --set overrides. "
            "Intended for --exp online."
        ),
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Override config values: key=value (space-separated). Example: --set loso_lr=1e-4 loso_epochs=60 subjects=1,2,3",
    )

    args = parser.parse_args()

    cfg = Config()

    # Apply preset ablation first (then allow --set to refine further)
    if args.ablation is not None:
        cfg = apply_overrides(cfg, ABLATION_CASES[args.ablation])

        # If user didn't choose a custom tag, use the ablation name
        if args.tag == "run":
            args.tag = f"ablation_{args.ablation}"

    overrides = _parse_set_args(args.set)
    cfg = apply_overrides(cfg, overrides)

    root_out = make_root_out(cfg, args.tag)

    if args.exp in ("online", "all"):
        run_online(cfg, root_out)
    if args.exp in ("offline", "all"):
        run_offline(cfg, root_out)
    if args.exp in ("loso", "all"):
        run_loso(cfg, root_out)

    print("Saved all outputs to:", root_out)


if __name__ == "__main__":
    main()
