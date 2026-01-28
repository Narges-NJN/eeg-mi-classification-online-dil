# models.py
import torch
from braindecode.models import EEGNet
from config import Config


def build_model(cfg: Config, n_chans: int, n_times: int, n_classes: int) -> torch.nn.Module:
    # EEGNet (same family as your better code request)
    return EEGNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length="auto",
        drop_prob=cfg.drop_prob,
    )
