import os
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

from helpers.configuration_container import ConfigurationContainer


def to_pytorch_variable(x):
    if is_cuda_enabled():
        x = x.cuda()
    return Variable(x)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def noise(batch_size, data_size):
    """
    Returns a variable with the dimensions (batch_size, data_size containing gaussian noise
    """
    shape = (batch_size,) + data_size if isinstance(data_size, tuple) else (batch_size, data_size)
    return to_pytorch_variable(torch.randn(shape))


def is_cuda_enabled():
    cc = ConfigurationContainer.instance()
    return is_cuda_available() and cc.settings["trainer"]["params"]["score"]["cuda"]


def is_cuda_available():
    return torch.cuda.is_available()


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length)) for start, length in zip(splits, split_sizes))


def calculate_net_weights_dist(net1, net2):
    """Calculate the L2 distance of two set of neural network weights

    Arguments:
        net1: sequential pytorch model
        net2: sequential pytorch model

    Return:
        l2_dist: float
    """
    l2_dist = 0
    for net1_layer_weights, net2_layer_weights in zip(net1.parameters(), net2.parameters()):
        l2_dist += torch.sum((net1_layer_weights - net2_layer_weights) ** 2)
    result = torch.sqrt(l2_dist).data.cpu().numpy()
    if result is np.ndarray:
        return result[0]
    else:
        return result
