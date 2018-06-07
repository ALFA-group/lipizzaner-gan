import os
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

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
    shape = (batch_size,) + data_size if isinstance(data_size, tuple) else (
        batch_size, data_size)
    return to_pytorch_variable(torch.randn(shape))


def is_cuda_enabled():
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

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))