from torchvision import datasets
from data.data_loader import DataLoader


class MNISTFashionDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.FashionMNIST, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 784

    @property
    def num_classes(self):
        return 10
