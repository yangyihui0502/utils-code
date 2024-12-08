from typing import Union, List, Dict
import numpy as np
import torch
from pathlib import Path


def Normalize(X, N):
    mean = N[0]
    std = N[1]
    return (X - mean) / std

def Renormalize(X, N):
    mean = N[0]
    std = N[1]
    return (X * std) + mean

def save_norm_data(norm_file, norm_data):
    Path(norm_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(norm_file, norm_data)


def to_cpu(batch, non_blocking=False, ignore_list: bool = False) -> torch.Tensor:
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_cpu(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dict({k: to_cpu(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking)
    else:  # numpy and others
        batch = torch.as_tensor(batch, device="cpu")
    return batch


def to_numpy(batch, non_blocking=False, ignore_list: bool = False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_numpy(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dict({k: to_numpy(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.asarray(batch)
    return batch

def Lerp(a, b, t):
    '''
        t: weight of b
    '''
    return a + (b-a) * t