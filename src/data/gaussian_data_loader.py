import torch
from torch.utils.data import Dataset
from data.data_loader import DataLoader

import matplotlib

matplotlib.use("Agg")
import numpy
import numpy as np
import scipy.stats as stats

import matplotlib.pylab as plt
from helpers.configuration_container import ConfigurationContainer

from sklearn.preprocessing import MinMaxScaler


N_RECORDS = 10000
N_VALUES_PER_RECORD = 1000
MEAN = 5
STD = 2


class GaussianDataLoader(DataLoader):
    """
    A simple dataloader that returns samples from a normal distribution
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(GaussianDataSet, use_batch, batch_size, n_batches, shuffle)
        self.cc = ConfigurationContainer.instance()
        self.data_size = self.cc.settings["dataloader"].get("data_size", N_VALUES_PER_RECORD)

    @property
    def n_input_neurons(self):
        return self.data_size

    @property
    def num_classes(self):
        return None


class GaussianDataSet(Dataset):
    def __init__(self, **kwargs):
        # self.data = 2+torch.randn(N_RECORDS, N_VALUES_PER_RECORD)
        self.cc = ConfigurationContainer.instance()
        self.mean = self.cc.settings["dataloader"].get("mean", MEAN)
        self.std = self.cc.settings["dataloader"].get("std", STD)

        self.number_of_records = self.cc.settings["dataloader"].get("number_of_records", N_RECORDS)
        batch_size = self.cc.settings["dataloader"].get("batch_size", 0)
        self.number_of_records = (
            int(self.number_of_records / batch_size) * batch_size
        )  # Correct number of records to fill batches

        self.data_size = self.cc.settings["dataloader"].get("data_size", N_VALUES_PER_RECORD)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_samples = numpy.random.normal(
            loc=self.mean, scale=self.std, size=(self.number_of_records, self.data_size)
        )

        self.data = torch.from_numpy(dataset_samples).float().to(device)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return self.number_of_records

    def save_images(self, tensor, filename, discriminator=None):
        plt.interactive(False)
        fig, ax = plt.subplots(nrows=5, ncols=5)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.001, hspace=0.001)
        x_min, x_max = self.mean - 3 * self.std, self.mean + 3 * self.std
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i][j].hist(self.data[(i + j) % self.data_size].data.numpy(), bins=50)
                ax[i][j].hist(tensor[i + j].data.numpy(), bins=50)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].set_xlim([x_min, x_max])
        plt.savefig(filename)
        plt.close()

    @property
    def points(modes):
        return [0]
