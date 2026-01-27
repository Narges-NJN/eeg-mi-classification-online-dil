import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_run_dir(base_dir: str, run_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_dir, f"{run_name}_{ts}")
    ensure_dir(out)
    return out

def _to_serializable(obj: Any):
    """Recursively convert numpy types to Python types for JSON serialization."""
    # numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # dicts
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    # lists / tuples
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    # everything else is returned as-is (must be JSON-serializable)
    return obj


def save_json(obj: Dict[str, Any], path: str):
    obj_clean = _to_serializable(obj)
    with open(path, "w") as f:
        json.dump(obj_clean, f, indent=2)

def torch_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
