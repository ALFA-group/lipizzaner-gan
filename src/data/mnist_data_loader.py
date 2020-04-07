from torchvision import datasets, transforms
from data.data_loader import DataLoader

from torchvision.utils import save_image
from helpers.pytorch_helpers import denorm

class MNISTDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.MNIST, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 784

    @property
    def num_classes(self):
        return 10

    def transform(self):
        if self.cc.settings['network']['name'] == 'ssgan_convolutional_mnist':
            return transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5)
                    )
                 ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5)
                    )
                ]
            )

    def save_images(self, images, shape, filename):
        if self.cc.settings['network']['name'] == 'ssgan_convolutional_mnist':
            img_view = images
        else:
            dimensions = 1 if len(shape) == 3 else shape[3]
            img_view = images.view(images.size(0), dimensions, shape[1], shape[2])
        save_image(denorm(img_view.data), filename)

    def transpose_data(self, data):
        if self.cc.settings['network']['name'] == 'ssgan_convolutional_mnist':
            return data
        else:
            return data.view(self.batch_size, -1)
