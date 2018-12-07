import os
from abc import ABC, abstractmethod

import torch
import torch.utils.data
from torchvision.transforms import transforms
from torchvision.utils import save_image

from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import denorm


class DataLoader(ABC):
    """
    Abstract base class for all dataloaders, cannot be instanced.
    """

    def __init__(self, dataset, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        """
        :param dataset: Dataset from torchvision.datasets.*, e.g. MNIST or CIFAR10
        :param use_batch: If set to False, all data records will be returned (without mini-batching). Read from config if set there.
        :param batch_size: Ignored if use_batch is set to False. Read from config if set there.
        :param n_batches: Number of batches to process per iteration. If set to 0, all batches are used. Read from config if set there.
        :param shuffle: Determines if the dataset will be shuffled each time samples are selected. Read from config if set there.
        :param max_size: Maximum amount of records selected from the dataset. Read from config if set there.
        """
        self.dataset = dataset
        self.cc = ConfigurationContainer.instance()
        settings = self.cc.settings['dataloader']
        self.use_batch = settings.get('use_batch', use_batch)
        self.batch_size = settings.get('batch_size', batch_size)
        self.n_batches = settings.get('n_batches', n_batches)
        self.shuffle = settings.get('shuffle', shuffle)

    def load(self):
        # Image processing

        # Dataset
        dataset = self.dataset(root=os.path.join(self.cc.settings['general']['output_dir'], 'data'),
                               train=True,
                               transform=self.transform(),
                               download=True)
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=self.batch_size if self.use_batch else len(dataset),
                                           shuffle=self.shuffle,
                                           num_workers=self.cc.settings['general']['num_workers'])

    def transform(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                               std=(0.5, 0.5, 0.5))])

    def transpose_data(self, data):
        return data.view(self.batch_size, -1)

    def save_images(self, images, shape, filename):
        # Additional dimensions are only passed to the shape instance when > 1
        dimensions = 1 if len(shape) == 3 else shape[3]

        img_view = images.view(images.size(0), dimensions, shape[1], shape[2])
        save_image(denorm(img_view.data), filename)

    @property
    @abstractmethod
    def n_input_neurons(self):
        pass
