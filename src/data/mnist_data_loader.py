from torchvision import datasets
from torchvision.transforms import transforms
from data.data_loader import DataLoader


class MNISTDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.MNIST, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 784

    def transform(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,),
                                                                               std=(0.5,))])

