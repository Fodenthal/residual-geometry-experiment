import os

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return best available accelerator (cuda, then mps), else cpu."""
    forced = os.environ.get("CODEC_EXPERIMENT_DEVICE")
    if forced:
        return torch.device(forced)

    if not prefer_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_available()
        and mps_backend.is_built()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def move_to_device(tensor_or_model, device: torch.device):
    """Move a tensor or model to the specified device and return it."""
    return tensor_or_model.to(device)
