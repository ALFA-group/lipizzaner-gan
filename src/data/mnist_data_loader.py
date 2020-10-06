import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from data.data_loader import DataLoader
from helpers.pytorch_helpers import denorm
from data.balanced_labels_batch_sampler import BalancedLabelsBatchSampler


class MNISTDataLoader(DataLoader):
    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.MNIST, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 784

    @property
    def num_classes(self):
        if self.cc.settings["dataloader"]["learning_type"] == "unsupervised":
            return 0
        else:
            return 10

    def load(self, train=True):
        label_rate = self.cc.settings["dataloader"].get("label_rate", None)
        if label_rate is None:
            return super().load(train=train)
        else:
            dataset = self.dataset(
                root=os.path.join(self.cc.settings["general"]["output_dir"], "data"),
                train=train,
                transform=self.transform(),
                download=True,
            )

            balanced_batch_sampler = BalancedLabelsBatchSampler(dataset, self.num_classes, self.batch_size, label_rate)
            return torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=self.cc.settings["general"]["num_workers"],
                batch_sampler=balanced_batch_sampler,
            )

    def transform(self):
        if self.cc.settings["network"]["name"] == "ssgan_convolutional_mnist":
            return transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            return super().transform()

    def save_images(self, images, shape, filename):
        if self.cc.settings["network"]["name"] == "ssgan_convolutional_mnist":
            img_view = images
            save_image(denorm(img_view.data), filename)
        else:
            super().save_images(images, shape, filename)

    def transpose_data(self, data):
        if self.cc.settings["network"]["name"] == "ssgan_convolutional_mnist":
            return data
        else:
            return super().transpose_data(data)
