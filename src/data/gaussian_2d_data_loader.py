import random

import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

from helpers.pytorch_helpers import to_pytorch_variable
from helpers.configuration_container import ConfigurationContainer
from data.data_loader import DataLoader

N_RECORDS = 10000
N_MODES = 12



class CircularToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toy problem 2d gaussian distributions of points in a circle
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(CircularToyDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 2

    @property
    def num_classes(self):
        return None

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)


class GridToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toy problem 2d gaussian distributions of points in a grid
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(GridToyDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 2

    @property
    def num_classes(self):
        return None

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)

    @property
    def points(self):
        pass

class Gaussian2DDataSet(Dataset):

    def __init__(self, **kwargs):
        self.cc = ConfigurationContainer.instance()
        self.number_of_records = self.cc.settings['dataloader'].get('number_of_records', N_RECORDS)
        number_of_modes = self.cc.settings['dataloader'].get('number_of_modes', N_MODES)

        number_of_modes = N_MODES if number_of_modes <= 0 else number_of_modes
        self.number_of_records = N_RECORDS if self.number_of_records <= 0 else self.number_of_records

        self.cc.settings['dataloader']['number_of_modes'] = number_of_modes #If does not exists it creates the parameter
        xs, ys = self.points(number_of_modes)
        points_array = np.array((xs, ys), dtype=np.float).T
        self.data = torch.from_numpy(np.random.normal(points_array[np.random.choice(points_array.shape[0], self.number_of_records), :], 0.25)).float()

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return self.number_of_records

    def save_images(self, tensor, filename, discriminator=None):
        plt.interactive(False)
        if not isinstance(tensor, list):
            plt.style.use('ggplot')
            plt.clf()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            data = self.data.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            ax1.scatter(x, y, c='lime', s=1)

            data = tensor.data.cpu().numpy() if hasattr(tensor, 'data') else tensor.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            ax1.scatter(x, y, c='red', marker='.', s=1)
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
                        ax.fill([x, x + 0.25, x + 0.25, x], [y, y, y + 0.25, y + 0.25], 'r', alpha=normalized[k],
                                zorder=0)
                        k += 1


class CircularToyDataSet(Gaussian2DDataSet):

    @staticmethod
    def points(number_of_modes):
        thetas = np.linspace(0, 2 * np.pi, number_of_modes+1)[:-1]
        return np.sin(thetas) * 10, np.cos(thetas) * 10


class GridToyDataSet(Gaussian2DDataSet):

    @staticmethod
    def points(number_of_modes):
        points_per_row = int(sqrt(number_of_modes))
        points_per_col = int(number_of_modes/ points_per_row)
        size = 5

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
