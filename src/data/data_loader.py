import os
from abc import ABC, abstractmethod

import logging
import torch
import torch.utils.data
from torchvision.transforms import transforms
from torchvision.utils import save_image

from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import denorm

from torch.utils.data import Sampler, SubsetRandomSampler
from helpers.reproducible_helpers import set_random_seed
import random

_logger = logging.getLogger()

class DataLoader(ABC):

    """
    Abstract base class for all dataloaders, cannot be instanced.
    """

    def __init__(
        self,
        dataset,
        use_batch=True,
        batch_size=100,
        n_batches=0,
        shuffle=False,
        sampling_ratio=1,
    ):
        """
        :param dataset: Dataset from torchvision.datasets.*, e.g. MNIST or CIFAR10
        :param use_batch: If set to False, all data records will be returned (without mini-batching). Read from config if set there.
        :param batch_size: Ignored if use_batch is set to False. Read from config if set there.
        :param n_batches: Number of batches to process per iteration. If set to 0, all batches are used. Read from config if set there.
        :param shuffle: Determines if the dataset will be shuffled each time samples are selected. Read from config if set there.
        :param sampling_ratio: Percentage in terms of ratio [0, 1] of the training data to be loaded to train the networks
        """
        self.dataset = dataset
        self.cc = ConfigurationContainer.instance()
        settings = self.cc.settings["dataloader"]
        self.dataset_name = settings.get("dataset_name", use_batch)
        self.use_batch = settings.get("use_batch", use_batch)
        self.batch_size = settings.get("batch_size", batch_size)
        self.n_batches = settings.get("n_batches", n_batches)
        self.shuffle = settings.get("shuffle", shuffle)
        self.sampling_ratio = settings.get("sampling_ratio", sampling_ratio)
        cell_settings = self.cc.settings["general"]["distribution"]
        self.cell_number = cell_settings.get("client_id", 0)

    def load(self, train=True):

        # Dataset
        split_param_keyword = "split" if self.dataset_name == "svhn" else "train"
        dataset_params = {
            "root": os.path.join(self.cc.settings["general"]["output_dir"], "data"),
            split_param_keyword: train,
            "transform": self.transform(),
            "download": True,
        }
        dataset = self.dataset(**dataset_params)

        if self.sampling_ratio >= 1:
            self.sampler = None
        else:
            set_random_seed(
                self.cc.settings["general"]["seed"] + self.cell_number,
                self.cc.settings["trainer"]["params"]["score"]["cuda"],
            )
            dataset_size = len(dataset)
            sample_size = int(dataset_size * self.sampling_ratio)
            mask = random.sample(range(dataset_size), sample_size)

            self.sampler = torch.utils.data.SubsetRandomSampler(mask)
            self.shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if self.use_batch else len(dataset),
            shuffle=self.shuffle,
            num_workers=self.cc.settings["general"]["num_workers"],
            sampler=self.sampler,
        )

    def transform(self):
        # return super.transform()
        return transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
         ])

    def transpose_data(self, data):
        return data.view(self.batch_size, -1)

    def save_images(self, images, shape, filename):
        # Additional dimensions are only passed to the shape instance when > 1
        if shape is not None:
            dimensions = 1 if len(shape) == 3 else shape[3]
            img_view = images.view(images.size(0), dimensions, shape[1], shape[2])
            save_image(denorm(img_view.data), filename)
        else:
            save_image(images.data, filename)

    @property
    @abstractmethod
    def n_input_neurons(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass
