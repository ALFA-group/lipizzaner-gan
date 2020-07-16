import os

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from data.data_loader import DataLoader
from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import denorm

from data.balanced_labels_batch_sampler import BalancedLabelsBatchSampler


class CIFAR10DataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.CIFAR10, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 3072

    @property
    def num_classes(self):
        return 10

    def load(self, train=True):
        label_rate = self.cc.settings['dataloader'].get('label_rate', None)
        if label_rate is None:
            return super().load(train=train)
        else:
            dataset = self.dataset(root=os.path.join(self.cc.settings['general']['output_dir'], 'data'),
                                   train=train,
                                   transform=self.transform(),
                                   download=True)

            balanced_batch_sampler = BalancedLabelsBatchSampler(
                dataset,
                self.num_classes,
                self.batch_size,
                label_rate
            )
            return torch.utils.data.DataLoader(dataset=dataset,
                                               num_workers=self.cc.settings['general']['num_workers'],
                                               batch_sampler=balanced_batch_sampler)

    def transform(self):
        return transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def save_images(self, images, shape, filename):
        save_image(denorm(images.data), filename)

    def transpose_data(self, data):
        return data