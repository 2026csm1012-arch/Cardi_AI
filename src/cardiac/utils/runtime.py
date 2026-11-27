import os
import platform
import random
import numpy as np
import torch


def setup(seed=2026, requested='auto'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Select CPU or enable a Colab/Kaggle GPU.')
    device = torch.device(requested)
    memory = torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None
    return device, {'device': str(device), 'gpu': torch.cuda.get_device_name(device) if memory else None,
                    'gpu_total_bytes': memory, 'torch': str(torch.__version__), 'python': platform.python_version(),
                    'batch_size': (4 if memory and memory >= 20 * 1024**3 else 2) if memory else 1,
                    'workers': min(2, os.cpu_count() or 1) if memory else 0,
                    'amp': device.type == 'cuda', 'seed': seed}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
