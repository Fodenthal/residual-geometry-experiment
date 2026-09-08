import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for torch, numpy, random, and torch.cuda for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng(seed: int) -> np.random.Generator:
    """Return a seeded numpy Generator."""
    return np.random.default_rng(seed)
