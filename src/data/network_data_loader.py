import random
import sys

import torch
from torch.utils.data import Dataset
import matplotlib

from helpers.pytorch_helpers import to_pytorch_variable

matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
sys.path.append('../')
sys.path.append("./network_data")
from data.data_loader import DataLoader


N_RECORDS = 42
SEQUENCE_LENGTH = 30
N_VALUES_PER_RECORD = 4


flow_data = np.load("./data/network_data/network_capture.npy")

def generate_random_sequences(num_sequences):
    sequences = []
    for i in range(num_sequences):
        StartTime = [i for i in range(SEQUENCE_LENGTH)]
        PktSize = [random.choice([j for j in range(400)]) for i in range(SEQUENCE_LENGTH)]
        SrcAddr = [random.choice([j for j in range(7)]) for i in range(SEQUENCE_LENGTH)]
        DstAddr = [random.choice([j for j in range(7)]) for i in range(SEQUENCE_LENGTH)]
        sequences.append(np.array((StartTime, PktSize, SrcAddr, DstAddr)).T)
    return torch.from_numpy(np.array(sequences))

class NetworkDataLoader(DataLoader):
    """
    A dataloader that returns network traffic data
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        self.use_batch = use_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.shuffle = shuffle
        self.sequence_length = SEQUENCE_LENGTH
        super().__init__(NetworkDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return N_VALUES_PER_RECORD

    def create_copy(self):
        return NetworkDataLoader(use_batch = self.use_batch, batch_size = self.batch_size, n_batches = self.n_batches, shuffle = self.shuffle)

    # Will probably need to define something different for this
    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)


class NetworkDataSet(Dataset):

    def __init__(self, **kwargs):
        self.data = flow_data
        print("Packets Array Size: ", self.data.shape)
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return N_RECORDS

    def save_images(self, tensor, filename, discriminator=None):
        # Part of old DataLoader
        pass

    @staticmethod
    def _plot_discriminator(discriminator, ax):
        # Part of old DataLoader
        pass
