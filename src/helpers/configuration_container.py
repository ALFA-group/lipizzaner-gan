import logging
import os

import sys

from helpers.singleton import Singleton
import importlib


@Singleton
class ConfigurationContainer:
    class_maps = {
        "bceloss": ("torch.nn", "BCELoss"),
        "mseloss": ("torch.nn", "MSELoss"),
        "celoss": ("torch.nn", "CrossEntropyLoss"),
        "heuristicloss": (
            "networks.customized_loss.heuristic_loss",
            "HeuristicLoss",
        ),
        "mustangs": ("networks.customized_loss.mustangs_loss", "MustangsLoss"),
        "mnist": ("data.mnist_data_loader", "MNISTDataLoader"),
        "mnist_fashion": (
            "data.mnist_fashion_data_loader",
            "MNISTFashionDataLoader",
        ),
        "covid-unsupervised": ("data.covid_data_loader", "CovidDataLoader"),
        "cifar": ("data.cifar10_data_loader", "CIFAR10DataLoader"),
        "svhn": ("data.svhn_data_loader", "SVHNDataLoader"),
        "celeba": ("data.celeba_data_loader", "CelebADataLoader"),
        "network_traffic": ("data.network_data_loader", "NetworkDataLoader"),
        "gaussian": ("data.gaussian_data_loader", "GaussianDataLoader"),
        "labeled_gaussian_circle": (
            "data.gaussian_2d_data_loader",
            "LabeledCircularToyDataLoader",
        ),
        "labeled_gaussian_grid": ("data.gaussian_2d_data_loader", "LabeledGridToyDataLoader"),
        "unlabeled_gaussian_circle": (
            "data.gaussian_2d_data_loader",
            "UnlabeledCircularToyDataLoader",
        ),
        "unlabeled_gaussian_grid": ("data.gaussian_2d_data_loader", "UnlabeledGridToyDataLoader"),
        "mooc": ("data.mooc_data_loader", "MOOCDataLoader"),
        "mooc_file": ("data.mooc_data_loader", "MOOCFileDataLoader"),
        "backprop": (
            "training.backpropagation_trainer",
            "BackpropagationTrainer",
        ),
        "sequential_nes": (
            "training.nes.sequential_nes_trainer",
            "SequentialNESTrainer",
        ),
        "parallel_nes": (
            "training.nes.parallel_nes_trainer",
            "ParallelNESTrainer",
        ),
        "alternating_ea": (
            "training.ea.alternating_ea_trainer",
            "AlternatingEATrainer",
        ),
        "parallel_ea": (
            "training.ea.parallel_ea_trainer",
            "ParallelEATrainer",
        ),
        "four_layer_perceptron": (
            "networks.network_factory",
            "FourLayerPerceptronFactory",
        ),
        "five_layer_perceptron": (
            "networks.network_factory",
            "FiveLayerPerceptronFactory",
        ),
        "conv_mnist_unsupervised": (
            "networks.network_factory",
            "ConvolutionalMNISTUnsupervised",
        ),
        "ssgan_perceptron": (
            "networks.network_factory",
            "SSGANFourLayerPerceptronFactory",
        ),
        "ssgan_svhn": (
            "networks.network_factory",
            "SSGANPerceptronSVHNFactory",
        ),
        "ssgan_convolutional": (
            "networks.network_factory",
            "SSGANConvolutionalNetworkFactory",
        ),
        "ssgan_convolutional_mnist": (
            "networks.network_factory",
            "SSGANConvolutionalMNISTNetworkFactory",
        ),
        "ssgan_conv_mnist_28x28": (
            "networks.network_factory",
            "SSGANConvMNIST28x28NetworkFactory",
        ),
        "convolutional_grayscale128x128": (
            "networks.network_factory",
            "ConvolutionalGrayscale128x128",
        ),
        "convolutional": (
            "networks.network_factory",
            "ConvolutionalNetworkFactory",
        ),
        "mooc_net": ("networks.mooc_net", "MOOCFourLayerPerceptronFactory"),
        "gaussian_2d_perceptron": (
            "networks.network_factory",
            "Gaussian2DNetworkFactory",
        ),
        "rnn": ("networks.network_factory", "RNNFactory"),
        "three_layer_perceptron": (
            "networks.network_factory",
            "ThreeLayerPerceptronFactory",
        ),
        "conditional_gaussian_2d_perceptron": (
            "networks.network_factory",
            "Gaussian2DConditionalNetworkFactory",
        ),
        "lipizzaner_gan": (
            "training.ea.lipizzaner_gan_trainer",
            "LipizzanerGANTrainer",
        ),
        "lipizzaner_wgan": (
            "training.ea.lipizzaner_wgan_trainer",
            "LipizzanerWGANTrainer",
        ),
    }

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.settings = {}
        self._output_dir = None

    def create_instance(self, name, *args, **kwargs):
        module_name, class_name = self.class_maps[name]
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(*args, **kwargs)

    # Encapsulated properties for often used settings
    @property
    def is_losswise_enabled(self):
        """
        :return: true if losswise sections exist and status is set to enabled
        """
        return (
            "losswise" in self.settings["general"]
            and "enabled" in self.settings["general"]["losswise"]
            and self.settings["general"]["losswise"]["enabled"]
        )

    @property
    def output_dir(self):
        """
        Also creates the output directory if it does not yet exists.
        :return: The output directoy specified in config file, combined with a method-specific subfolder
        """

        if self._output_dir is None:
            self._output_dir = self._load_output_dir()
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    def _load_output_dir(self):
        output = self.settings["general"]["output_dir"] if "output_dir" in self.settings["general"] else "output"
        subdir = (
            self.settings["trainer"]["method"]["name"]
            if "method" in self.settings["trainer"]
            else self.settings["trainer"]["name"]
        )
        directory = os.path.join(output, subdir)

        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory
