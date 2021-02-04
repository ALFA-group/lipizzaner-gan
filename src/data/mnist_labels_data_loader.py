from torchvision import datasets
from data.data_loader import DataLoader
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import torch.utils.data
import os


class MNISTLabelsDataLoader(DataLoader):

    def __init__(self, labels, num_labels_per_cell, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        # start by just choosing one label randomly from the list
        if labels:
            self.labels = random.sample(labels, min(num_labels_per_cell, len(labels)))
        else:
            self.labels = random.sample(range(9))
        # get indices corresponding to chosen label and sample with those
        self.indices = [i for (i, x) in  enumerate(datasets.MNIST('mnist_dataset', download=True).train_labels) if x in self.labels]
        self.dataset = datasets.MNIST
        super().__init__(datasets.MNIST, use_batch, batch_size, n_batches, shuffle)

    def load(self):
        # Image processing
        
        # Dataset
        dataset = self.dataset(root=os.path.join(self.cc.settings['general']['output_dir'], 'data'),
                               train=True,
                               transform=self.transform(),
                               download=True)
        
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=self.batch_size if self.use_batch else len(dataset),
                                           drop_last = True,
                                           shuffle=False,
                                           num_workers=1,
                                           sampler=SubsetRandomSampler(self.indices))

        
    @property
    def n_input_neurons(self):
        return 784

    def transform(self):
        return transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
        ])

