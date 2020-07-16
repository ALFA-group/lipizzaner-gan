from networks.standalone_sgan.data_loader import DataLoader
from helpers.pytorch_helpers import denorm
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CIFAR10DataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.CIFAR10, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 3072

    @property
    def num_classes(self):
        return 10

    def transform(self):
        return transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def save_images(self, images, shape, filename):
        save_image(denorm(images.data), filename)

    def transpose_data(self, data):
        return data