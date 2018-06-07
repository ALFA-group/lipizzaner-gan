import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data.data_loader import DataLoader


class MOOCDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super(MOOCDataLoader, self).__init__(MOOCDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return MOOCDataSet.N_DIMENSIONS_PER_EXEMPLAR

    def save_images(self, images, shape, filename):
        filename = "{}.csv".format(os.path.splitext(filename)[0])
        np.savetxt(filename, images.data.numpy(), delimiter=',')

    
class MOOCFileDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super(MOOCFileDataLoader, self).__init__(MOOCDataFile, use_batch, batch_size, n_batches, shuffle)

        self.filename = self.cc.settings['dataloader'].get('file_name', None)
        assert os.path.exists(self.filename)
        self.dimensions = np.genfromtxt(self.filename, delimiter=',', max_rows=1).shape[0]
            
    @property
    def n_input_neurons(self):
        return self.dimensions

    def save_images(self, images, shape, filename):
        filename = "{}.csv".format(os.path.splitext(filename)[0])
        np.savetxt(filename, images.data.numpy(), delimiter=',')

    def load(self):
        # Dataset
        dataset = self.dataset(filename=self.filename, root=os.path.join(self.cc.settings['general']['output_dir'], 'data'),
                               train=True,
                               transform=self.transform(),
                               download=True)
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=self.batch_size if self.use_batch else len(dataset),
                                           shuffle=self.shuffle)

    
class MOOCDataSet(Dataset):
    N_EXEMPLARS = 200
    N_DIMENSIONS_PER_EXEMPLAR = 3

    def __init__(self, **kwargs):
        xs = self.points()
        self.data = torch.from_numpy(xs[np.random.choice(xs.shape[0], MOOCDataSet.N_EXEMPLARS), :]).float()

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return MOOCDataSet.N_EXEMPLARS

    @staticmethod
    def points():
        MUS = [90, 500, 70]
        SIGMAS = np.array([[2, -1, 0], [-1, 2, 1], [0, 1, 2]]) * 10
        # Draw grades
        xs = np.random.multivariate_normal(mean=MUS, cov=SIGMAS, size=MOOCDataSet.N_EXEMPLARS)
        xs[xs[:,0] > 100, 0] = 100
        xs[:, 1].astype(int)
        xs[:, 2].astype(int)
        return xs


class MOOCDataFile(Dataset):

    def __init__(self, **kwargs):
        self.data = torch.from_numpy(np.loadtxt(kwargs['filename'], delimiter=",")).float()

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return self.data.shape[0]
