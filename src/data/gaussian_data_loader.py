import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from data.data_loader import DataLoader

N_RECORDS = 10000
N_VALUES_PER_RECORD = 1000


class GaussianDataLoader(DataLoader):
    """
    A simple dataloader that returns samples from a normal distribution
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(GaussianDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return N_VALUES_PER_RECORD

    @property
    def num_classes(self):
        return None


class GaussianDataSet(Dataset):

    def __init__(self, **kwargs):
        self.data = torch.randn(N_RECORDS, N_VALUES_PER_RECORD)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return N_RECORDS

    @staticmethod
    def save_images(tensor, filename, discriminator=None):
        plt.interactive(False)
        fig, ax = plt.subplots(nrows=10, ncols=10)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.001, hspace=0.001)
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i][j].hist(tensor[i + j].data.numpy(), bins=50)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
        plt.savefig(filename)
