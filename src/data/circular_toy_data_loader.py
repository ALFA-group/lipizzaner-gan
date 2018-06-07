import numpy as np

from data.data_loader import DataLoader
from data.grid_toy_data_loader import GridToyDataSet

N_RECORDS = 50000
N_VALUES_PER_RECORD = 2
N_MODES = 12


class CircularToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toyproblem distribution of 5 points in a circle
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(CircularToyDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return N_VALUES_PER_RECORD

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)


class CircularToyDataSet(GridToyDataSet):

    @staticmethod
    def points():
        thetas = np.linspace(0, 2 * np.pi, N_MODES)
        return np.sin(thetas) * 4, np.cos(thetas) * 4
