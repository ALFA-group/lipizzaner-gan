import os
import importlib

from helpers.ignore_label_dataset import IgnoreLabelDataset
from torchvision import transforms
from networks.standalone_sgan.fid_score import FIDCalculator


class ScoreCalculatorFactory:
    @staticmethod
    def create_instance(*args, dataset):
        if dataset == "mnist":
            module_name, class_name = ("networks.standalone_sgan.mnist_data_loader", "MNISTDataLoader")
        elif dataset == "cifar10":
            module_name, class_name = (
                "networks.standalone_sgan.cifar10_data_loader",
                "CIFAR10DataLoader",
            )
        else:
            module_name, class_name = (None, None)

        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(*args)

    @staticmethod
    def create(dataset_name, cuda=False, score_sample_size=64):
        # Need to add a dataloader here
        dataloader = ScoreCalculatorFactory.create_instance(dataset=dataset_name)
        dataloader.load()

        transforms_op = [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        # CIFAR
        if dataset_name == "cifar10":
            transforms_op = [transforms.Resize([64, 64])] + transforms_op

        dataset = dataloader.dataset(
            root=os.path.join("./networks/standalone_sgan/output", "data"),
            train=True,
            transform=transforms.Compose(transforms_op),
        )

        return FIDCalculator(
            IgnoreLabelDataset(dataset), cuda=cuda, n_samples=score_sample_size, dataset_name=dataset_name
        )
