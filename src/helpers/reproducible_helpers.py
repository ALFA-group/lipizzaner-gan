import random

import numpy as np
import torch


def set_random_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_heuristic_seed(seed, ip, port):
    """
    Heuristic method to obtain seed number based on client IP and port
    (Since it is desired to have different seed for different clients to ensure diversity)
    """
    # TODO Handle the case of integer overflow
    seed + int(ip.replace('.', '')) + 1000*port
    seed = seed % (2**32 - 1)
    return seed
