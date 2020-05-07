import os

from torchvision import datasets, transforms
from torchvision.utils import save_image

from data.data_loader import DataLoader
from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import denorm


class CIFAR10DataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.CIFAR10, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 3072

    def transform(self):
        return transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def save_images(self, images, shape, filename):
        save_image(denorm(images.data), filename)

    '''def transpose_data(self, data):
        return data'''