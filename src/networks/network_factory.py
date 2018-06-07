from abc import ABC, abstractmethod

from torch import nn
from torch.nn import Sequential

from helpers.configuration_container import ConfigurationContainer
from networks.competetive_net import DiscriminatorNet, GeneratorNet


class NetworkFactory(ABC):

    def __init__(self, input_data_size, loss_function=None):
        """
        :param loss_function: The loss function computing the network error, e.g. BCELoss. Read from config if not set.
        :param input_data_size: The number of discriminator input/generator output neurons,
        e.g. 784 for MNIST and 3072 for CIFAR (provided by DataLoader instances)
        """
        cc = ConfigurationContainer.instance()
        if loss_function is None:
            self.loss_function = cc.create_instance(cc.settings['network']['loss'])
        else:
            self.loss_function = loss_function
        self.input_data_size = input_data_size

    @abstractmethod
    def create_generator(self, parameters=None):
        """
        :param parameters: The parameters for the neural network. If not set, default (random) values will be used
        :return: A neural network with 64 seed input neurons and input_data_size output neurons
        """
        pass

    @abstractmethod
    def create_discriminator(self, parameters=None):
        """
        :param parameters: The parameters for the neural network. If not set, default (random) values will be used
        :return: A neural network with input_data_size input neurons and one output (real/fake)
        """
        pass

    def create_both(self):
        """
        :return: A tuple containing the results of create_generator and create_discriminator (with default parameters)
        """
        return self.create_generator(), self.create_discriminator()

    @property
    @abstractmethod
    def gen_input_size(self):
        pass


class FourLayerPerceptronFactory(NetworkFactory):

    @property
    def gen_input_size(self):
        return 64

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = GeneratorNet(
            self.loss_function,
            Sequential(
                nn.Linear(64, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, self.input_data_size),
                nn.Tanh()), self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net

    def create_discriminator(self, parameters=None, encoded_parameters=None):
        net = DiscriminatorNet(
            self.loss_function,
            Sequential(
                nn.Linear(self.input_data_size, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()), self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net


class CircularProblemFactory(NetworkFactory):

    @property
    def gen_input_size(self):
        return 256

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = GeneratorNet(
            self.loss_function,
            Sequential(
                nn.Linear(self.gen_input_size, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, self.input_data_size)
            ), self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net

    def create_discriminator(self, parameters=None, encoded_parameters=None):
        net = DiscriminatorNet(
            self.loss_function,
            Sequential(
                nn.Linear(self.input_data_size, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
                nn.Sigmoid()), self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net


class ConvolutionalNetworkFactory(NetworkFactory):

    complexity = 64

    @property
    def gen_input_size(self):
        return 100, 1, 1

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = GeneratorNet(
            self.loss_function,
            nn.Sequential(
                nn.ConvTranspose2d(100, self.complexity * 8, 4, 1, 0),
                nn.BatchNorm2d(self.complexity * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.complexity * 8, self.complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.complexity * 4, self.complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.complexity * 2, self.complexity, 4, 2, 1),
                nn.BatchNorm2d(self.complexity),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.complexity, 3, 4, 2, 1),
                nn.Tanh()
            ),
            self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        elif encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters
        else:
            net.net.apply(self._init_weights)

        return net

    def create_discriminator(self, parameters=None, encoded_parameters=None):
        net = DiscriminatorNet(
            self.loss_function,
            Sequential(
                nn.Conv2d(3, self.complexity, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity, self.complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 2, self.complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 4, self.complexity * 8, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 8, 1, 4, 1, 0),
                nn.Sigmoid()
            ),
            self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        elif encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters
        else:
            net.net.apply(self._init_weights)

        return net

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()