# models.py
from __future__ import annotations

import inspect

import torch

from config import Config


def _build_eegnet(cfg: Config, n_chans: int, n_times: int, n_classes: int) -> torch.nn.Module:
    """Build EEGNet from braindecode, but stay compatible across versions.

    Different braindecode releases renamed some constructor arguments (e.g.,
    n_outputs vs n_classes) and not all versions expose every hyperparameter.
    We therefore:
      1) create a superset of kwargs we care about for ablations
      2) filter to only what the installed EEGNet accepts
      3) fill required output arg name based on signature
    """
    try:
        from braindecode.models import EEGNet  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "braindecode is required to build EEGNet. "
            "Install it (e.g., pip install braindecode) in your environment."
        ) from e

    sig = inspect.signature(EEGNet.__init__)
    params = set(sig.parameters.keys())

    # Handle output argument name across versions
    out_kwargs = {}
    if "n_outputs" in params:
        out_kwargs["n_outputs"] = n_classes
    elif "n_classes" in params:
        out_kwargs["n_classes"] = n_classes
    else:
        # Fall back to the previous default used in your repo.
        out_kwargs["n_outputs"] = n_classes

    # Superset of kwargs we may want to ablate.
    candidate_kwargs = {
        "n_chans": n_chans,
        "n_times": n_times,
        "final_conv_length": "auto",
        "drop_prob": cfg.drop_prob,
        "F1": cfg.F1,
        "D": cfg.D,
        "kernel_length": cfg.kernel_length,
        "depthwise_kernel_length": cfg.depthwise_kernel_length,
        "pool_mode": cfg.pool_mode,
        "batch_norm_momentum": cfg.batch_norm_momentum,
    }
    candidate_kwargs.update(out_kwargs)

    # Keep only what this EEGNet supports.
    filtered = {k: v for k, v in candidate_kwargs.items() if k in params}

    # If the signature doesn't expose an ablation arg, we silently ignore it.
    # That keeps the same CLI working across braindecode versions.
    return EEGNet(**filtered)


def build_model(cfg: Config, n_chans: int, n_times: int, n_classes: int) -> torch.nn.Module:
    return _build_eegnet(cfg, n_chans=n_chans, n_times=n_times, n_classes=n_classes)
