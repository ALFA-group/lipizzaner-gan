import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from data.data_loader import DataLoader
from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import denorm

from data.balanced_labels_batch_sampler import BalancedLabelsBatchSampler

from src.helpers.pytorch_helpers import to_pytorch_variable


class SVHNDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.SVHN, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 3072

    @property
    def num_classes(self):
        return 10

    def load(self, train=True):
        label_rate = self.cc.settings['dataloader'].get('label_rate', None)
        dataset_type = 'train' if train else 'test'
        if label_rate is None:
            return super().load(split=dataset_type)
        else:
            dataset = self.dataset(root=os.path.join(self.cc.settings['general']['output_dir'], 'data'),
                                   split=dataset_type,
                                   transform=self.transform(),
                                   download=True)

            labels_list = []
            for _, label in dataset:
                labels_list.append(label)

            # Create a dictionary from labels to all their indices in the
            # dataset
            labels = to_pytorch_variable(torch.LongTensor(labels_list))
            min_count = min(labels.bincount(minlength=self.num_classes)).item()

            # express min_count as the nearest multiple of 10
            min_count = int(min_count / 10) * 10

            labels_set = list(set(labels.data.cpu().numpy()))
            label_to_indices = {
                label: np.where(labels.data.cpu().numpy() == label)[0]
                for label in labels_set}

            # Shuffle the indices of each label
            indices = []
            for label in labels_set:
                np.random.shuffle(label_to_indices[label])
                indices.extend(label_to_indices[label][:min_count])

            balanced_subset = torch.utils.data.Subset(dataset, indices)

            balanced_batch_sampler = BalancedLabelsBatchSampler(
                balanced_subset,
                self.num_classes,
                self.batch_size,
                label_rate
            )
            return torch.utils.data.DataLoader(dataset=balanced_subset,
                                               num_workers=self.cc.settings['general']['num_workers'],
                                               batch_sampler=balanced_batch_sampler)

    def transform(self):
        return super().transform()

    def save_images(self, images, shape, filename):
        save_image(denorm(images.data), filename)

    def transpose_data(self, data):
        return data