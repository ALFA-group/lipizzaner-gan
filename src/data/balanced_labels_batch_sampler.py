import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from helpers.pytorch_helpers import to_pytorch_variable


class BalancedLabelsBatchSampler(BatchSampler):
    """
    BatchSampler - from an MNIST-like dataset, samples n_classes and within
    these classes samples (label_rate * batch_size) / n_classes number of
    samples.

    Returns batches of size batch_size
    """

    def __init__(self, dataset, num_classes, batch_size, label_rate):
        import logging
        _logger = logging.getLogger(__name__)
        _logger.info("Using Balanced Labels Batch Sampler")
        self.dataset = dataset
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.label_rate = label_rate
        self.num_labels_per_batch = int(self.label_rate * self.batch_size)
        # Ensure that the dataset can be broken down exactly into batch size
        assert len(self.dataset) % self.batch_size == 0

        # Load the dataset
        loader = DataLoader(dataset)
        # Get all the labels in this dataset
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        # Create a dictionary from labels to all their indices in the dataset
        self.labels = to_pytorch_variable(torch.LongTensor(self.labels_list))
        self.labels_set = list(set(self.labels.data.cpu().numpy()))
        self.label_to_indices = {label: np.where(self.labels.data.cpu().numpy() == label)[0]
                                 for label in self.labels_set}
        # Shuffle the indices of each label
        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])

        # Break down dataset into 'labeled' and 'unlabeled' datasets
        self.labeled_data = np.array([], dtype=int)
        self.unlabeled_data = np.array([], dtype=int)
        total_labeled_data = int(len(self.dataset) * self.label_rate)
        self.num_labels_per_class = int(total_labeled_data / self.num_classes)
        for label in self.labels_set:
            self.labeled_data = np.concatenate(
                (
                    self.labeled_data,
                    self.label_to_indices[label][:self.num_labels_per_class]
                )
            )
            self.unlabeled_data = np.concatenate(
                (
                    self.unlabeled_data,
                    self.label_to_indices[label][self.num_labels_per_class:]
                )
            )
        np.random.shuffle(self.labeled_data)
        np.random.shuffle(self.unlabeled_data)

    def __iter__(self):
        num_batches = self.__len__()
        for i in range(num_batches):
            indices = []
            # Get labeled data for this batch
            indices.extend(
                self.labeled_data[
                    i * self.num_labels_per_batch:
                    (i + 1) * self.num_labels_per_batch
                ]
            )
            # Get unlabeled data for this batch
            num_remaining_data_points = self.batch_size - self.num_labels_per_batch
            indices.extend(
                self.unlabeled_data[
                    i * num_remaining_data_points:
                    (i + 1) * num_remaining_data_points
                ]
            )
            yield indices

    def __len__(self):
        return len(self.dataset) // self.batch_size
