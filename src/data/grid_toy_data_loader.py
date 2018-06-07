import random

import torch
from torch.utils.data import Dataset
import matplotlib

from helpers.pytorch_helpers import to_pytorch_variable

matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

from data.data_loader import DataLoader

N_RECORDS = 50000
N_VALUES_PER_RECORD = 2
N_MODES = 12


class GridToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toyproblem distribution multiple fixed points in a grid
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(GridToyDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return N_VALUES_PER_RECORD

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)


class GridToyDataSet(Dataset):

    def __init__(self, **kwargs):
        xs, ys = self.points()
        points_array = np.array((xs, ys), dtype=np.float).T
        self.data = torch.from_numpy(points_array[np.random.choice(points_array.shape[0], N_RECORDS), :]).float()

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return N_RECORDS

    @staticmethod
    def points():
        points_per_row = 6
        points_per_col = 6
        size = 2

        incr_x = (size * 2) / (points_per_row - 1)
        incr_y = (size * 2) / (points_per_col - 1)
        xs = []
        ys = []
        for i in range(points_per_row):
            for j in range(points_per_col):
                xs.append((i * incr_x) - size)
                ys.append((j * incr_y) - size)
        return xs, ys

    colors = None

    def save_images(self, tensor, filename, discriminator=None):
        plt.interactive(False)
        if not isinstance(tensor, list):
            plt.style.use('ggplot')
            plt.clf()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            data = tensor.data.cpu().numpy() if hasattr(tensor, 'data') else tensor.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            for i in range(len(x)):
                rand_x = random.gauss(mu=x, sigma=0.1)
                rand_y = random.gauss(mu=y, sigma=0.1)
                ax1.scatter(rand_x, rand_y, c='red', marker='.', s=1)

            self._plot_discriminator(discriminator, ax1)

            x_original, y_original = self.points()
            ax1.scatter(x_original, y_original, c='lime')

        else:
            if GridToyDataSet.colors is None:
                GridToyDataSet.colors = [np.random.rand(3, ) for _ in tensor]

            plt.style.use('ggplot')
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            GridToyDataSet._plot_discriminator(discriminator, ax1)
            # Plot generator
            x_original, y_original = self.points()
            ax1.scatter(x_original, y_original, zorder=len(tensor) + 1, color='b')
            cm = plt.get_cmap('gist_rainbow')
            ax1.set_prop_cycle('color', [cm(1. * i / 10) for i in range(10)])
            for i, element in enumerate(tensor):
                data = element.data.cpu().numpy() if hasattr(element, 'data') else element.cpu().numpy()
                x, y = np.split(data, 2, axis=1)

                ax1.scatter(x.flatten(), y.flatten(), color=GridToyDataSet.colors[i],
                            zorder=len(tensor) - i, marker='x')

        plt.savefig(filename)

    @staticmethod
    def _plot_discriminator(discriminator, ax):
        if discriminator is not None:
            alphas = []
            for x in np.linspace(-1, 1, 8, endpoint=False):
                for y in np.linspace(-1, 1, 8, endpoint=False):
                    center = torch.zeros(2)
                    center[0] = x + 0.125
                    center[1] = y + 0.125
                    alphas.append(float(discriminator.net(to_pytorch_variable(center))))

            alphas = np.asarray(alphas)
            normalized = (alphas - min(alphas)) / (max(alphas) - min(alphas))
            plt.text(0.1, 0.9, 'Min: {}\nMax: {}'.format(min(alphas), max(alphas)), transform=ax.transAxes)

            k = 0
            for x in np.linspace(-1, 1, 8, endpoint=False):
                for y in np.linspace(-1, 1, 8, endpoint=False):
                    center = torch.zeros(2)
                    center[0] = x + 0.125
                    center[1] = y + 0.125
                    ax.fill([x, x + 0.25, x + 0.25, x], [y, y, y + 0.25, y + 0.25], 'r', alpha=normalized[k], zorder=0)
                    k += 1
