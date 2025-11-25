import torch

def get_device():
    """
    Selects the best available device for training.
    Priority: CUDA -> MPS -> CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_name(device):
    """
    Returns a string representation of the device.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    elif device.type == "mps":
        return "Apple MPS"
    else:
        return "CPU"
