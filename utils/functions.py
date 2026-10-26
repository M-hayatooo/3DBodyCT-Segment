import random

import numpy as np
import torch


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def write_fig_one(path, values, label="train"):
    with open(path, "w") as f:
        for value in values:
            f.write(f"{label}={value}\n")
    return
