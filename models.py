import torch
from braindecode.models import EEGNetv4

def make_eegnet(n_chans: int, n_times: int, n_classes: int, drop_prob: float = 0.0) -> torch.nn.Module:
    # EEGNetv4 expects input (batch, chans, times) by default in braindecode
    # It outputs logits (batch, n_classes)
    model = EEGNetv4(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length="auto",
        drop_prob=drop_prob
    )
    return model
