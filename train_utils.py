from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = EEGDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    losses = []
    all_y = []
    all_pred = []
    all_prob = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        losses.append(loss.item())
        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_prob.append(prob.detach().cpu().numpy())

    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)

    acc = float((pred == y).mean())
    return {"loss": float(np.mean(losses)), "acc": acc, "y": y, "pred": pred, "prob": prob}

def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, grad_clip: Optional[float]) -> Dict[str, float]:
    model.train()
    losses = []
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

    return {"loss": float(np.mean(losses)), "acc": float(correct / max(total, 1))}
