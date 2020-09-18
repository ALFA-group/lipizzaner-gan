import random
import torch
from torch.utils.data import Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
from helpers.pytorch_helpers import to_pytorch_variable
from helpers.configuration_container import ConfigurationContainer
from data.data_loader import DataLoader
from math import sqrt

N_RECORDS = 10000
N_MODES = 12
FACTOR_SIZE = 0.8


class LabeledCircularToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toy problem 2d gaussian distributions of points in a circle
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False, dataset_name=None):
        dataset = LabeledCircularToyDataSet if dataset_name is None else dataset_name
        super().__init__(dataset, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 2

    @property
    def num_classes(self):
        return self.dataset().num_classes

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)


class UnlabeledCircularToyDataLoader(LabeledCircularToyDataLoader):
    """
    A dataloader that returns samples from a simple toy problem 2d gaussian distributions of points in a circle
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(use_batch, batch_size, n_batches, shuffle, dataset_name=UnlabeledCircularToyDataSet)


class LabeledGridToyDataLoader(DataLoader):
    """
    A dataloader that returns samples from a simple toy problem 2d gaussian distributions of points in a grid
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False, dataset_name=None):
        dataset = LabeledGridToyDataSet if dataset_name is None else dataset_name
        super().__init__(dataset, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 2

    @property
    def num_classes(self):
        return self.dataset().num_classes

    def save_images(self, images, shape, filename):
        self.dataset().save_images(images, filename)

    @property
    def points(self):
        pass


class UnlabeledGridToyDataLoader(LabeledGridToyDataLoader):
    """
    A dataloader that returns samples from a simple toy problem 2d gaussian distributions of points in a grid
    """

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(use_batch, batch_size, n_batches, shuffle, dataset_name=UnlabeledGridToyDataSet)


class Gaussian2DDataSet(Dataset):
    def __init__(self, **kwargs):
        self.cc = ConfigurationContainer.instance()
        number_of_modes = self.cc.settings["dataloader"].get("number_of_modes", N_MODES)
        number_of_modes = N_MODES if number_of_modes <= 0 else number_of_modes
        self.cc.settings["dataloader"]["number_of_modes"] = number_of_modes  # If doesn't exist it creates the parameter

        self.number_of_records = self.cc.settings["dataloader"].get("number_of_records", N_RECORDS)
        self.number_of_records = N_RECORDS if self.number_of_records <= 0 else self.number_of_records

        xs, ys, labels = self.points(number_of_modes)
        points_with_label = np.array((xs, ys, labels), dtype=np.float).T
        points_with_label = points_with_label[np.random.choice(points_with_label.shape[0], self.number_of_records), :]
        points_array = [[point[0], point[1]] for point in points_with_label]

        self.labels = [int(point[2]) for point in points_with_label]
        self.data = torch.from_numpy(
            np.random.normal(
                points_array,
                0.025,
            )
        ).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.number_of_records

    def save_images(self, tensor, shape, filename):
        plt.interactive(False)
        if not isinstance(tensor, list):
            plt.style.use("ggplot")
            plt.clf()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            data = self.data.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            if self.num_classes > 0:
                colors = matplotlib.cm.rainbow(np.linspace(0, 1, self.num_classes))
                ax1.scatter(x, y, c=colors[self.labels], s=1)
            else:
                ax1.scatter(x, y, c="lime", s=1)
            data = tensor.data.cpu().numpy() if hasattr(tensor, "data") else tensor.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            ax1.scatter(x, y, c="red", marker=".", s=1)

        plt.savefig(filename)

    def save_images(self, tensor, filename, discriminator=None):
        plt.interactive(False)
        if not isinstance(tensor, list):
            plt.style.use("ggplot")
            plt.clf()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            data = self.data.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            if self.num_classes > 0:
                colors = matplotlib.cm.rainbow(np.linspace(0, 1, self.num_classes))
                ax1.scatter(x, y, c=colors[self.labels], s=1)
            else:
                ax1.scatter(x, y, c="lime", s=1)
            data = tensor.data.cpu().numpy() if hasattr(tensor, "data") else tensor.cpu().numpy()
            x, y = np.split(data, 2, axis=1)
            x = x.flatten()
            y = y.flatten()
            ax1.scatter(x, y, c="red", marker=".", s=1)

        plt.savefig(filename)

    @staticmethod
    def create_labels(number_of_modes):
        label_list = list()
        for label in range(number_of_modes):
            aux_list = [0] * number_of_modes
            aux_list[label] = 1
            label_list.append(aux_list)
        return label_list

    def create_label(self, label):
        aux_list = [0] * self.num_classes
        aux_list[label] = 1
        return aux_list

    @staticmethod
    def get_label_id(label):
        return label.index(1)

    @staticmethod
    def get_labels_id(labels_list):
        return [label.index(1) for label in labels_list]


class LabeledCircularToyDataSet(Gaussian2DDataSet):
    @property
    def num_classes(self):
        return self.cc.settings["dataloader"].get("number_of_modes", N_MODES)

    @staticmethod
    def points(number_of_modes):
        thetas = np.linspace(0, 2 * np.pi, number_of_modes + 1)[:-1]
        return np.sin(thetas) * FACTOR_SIZE, np.cos(thetas) * FACTOR_SIZE, list(range(number_of_modes))


class UnlabeledCircularToyDataSet(LabeledCircularToyDataSet):
    @property
    def num_classes(self):
        return 0


class LabeledGridToyDataSet(Gaussian2DDataSet):
    @property
    def num_classes(self):
        number_of_modes = self.cc.settings["dataloader"].get("number_of_modes", N_MODES)
        points_per_row = int(sqrt(number_of_modes))
        points_per_col = int(number_of_modes / points_per_row)
        return points_per_col * points_per_row

    @staticmethod
    def points(number_of_modes):
        points_per_row = int(sqrt(number_of_modes))
        points_per_col = int(number_of_modes / points_per_row)
        size = FACTOR_SIZE
        incr_x = (size * 2) / (points_per_row - 1)
        incr_y = (size * 2) / (points_per_col - 1)
        xs = []
        ys = []
        for i in range(points_per_row):
            for j in range(points_per_col):
                xs.append((i * incr_x) - size)
                ys.append((j * incr_y) - size)
        return xs, ys, list(range(len(xs)))


class UnlabeledGridToyDataSet(LabeledGridToyDataSet):
    @property
    def num_classes(self):
        return 0
