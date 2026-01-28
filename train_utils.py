# train_utils.py
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def train_one_epoch(model, dataloader, criterion, optimizer, device: str) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for Xb, yb in dataloader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

    return total_loss / max(1, len(dataloader)), correct / max(1, total)


@torch.no_grad()
def evaluate_full(model, dataloader, criterion, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ys, preds, probs = [], [], []

    for Xb, yb in dataloader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        logits = model(Xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item())

        p = F.softmax(logits, dim=1)
        pred = p.argmax(dim=1)

        ys.append(yb.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())

        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

    out = {
        "loss": np.array(total_loss / max(1, len(dataloader)), dtype=float),
        "acc": np.array(correct / max(1, total), dtype=float),
        "y": np.concatenate(ys) if ys else np.array([], dtype=int),
        "pred": np.concatenate(preds) if preds else np.array([], dtype=int),
        "prob": np.concatenate(probs) if probs else np.zeros((0, 2), dtype=float),
    }
    return out
