from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import Sequential
from torch.nn import RNN
from torch.autograd import Variable

from helpers.configuration_container import ConfigurationContainer
from networks.competetive_net import (
    DiscriminatorNet,
    GeneratorNet,
    GeneratorNetSequential,
    DiscriminatorNetSequential,
    SSDiscriminatorNet,
    SSGeneratorNet
)

from helpers.view_layer import View


class NetworkFactory(ABC):

    def __init__(self, input_data_size, num_classes=None, loss_function=None):
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

        self.num_classes = num_classes


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

class RNNFactory(NetworkFactory):
    @property
    def gen_input_size(self):
        return 10

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = GeneratorNetSequential(
            self.loss_function,
            SimpleRNN(self.gen_input_size, self.input_data_size, self.gen_input_size),
            self.gen_input_size
        )

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net

    def create_discriminator(self, parameters=None, encoded_parameters=None):
        net = DiscriminatorNetSequential(
            self.loss_function,
            SimpleRNN(self.input_data_size, 1, self.gen_input_size),
            self.gen_input_size
        )

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
                nn.Sigmoid()),
            self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net


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


class SimpleRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.inp = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden=None):
        """
        inputs: (batches, sequence_number, rnn_inputs)

        returns outputs of shape (batches, sequence_number, output_size)
        """
        samples = inputs.size(0)
        steps = inputs.size(1)
        outputs = Variable(torch.zeros(samples, steps, self.output_size))

        hidden = None
        for step in range(steps):
            # Get only one step of the function
            input = inputs[:,step,:].unsqueeze(1)

            # Go through inp layer
            intermediate = self.inp(input.float())

            # Go through RNN layer
            rnn_out, hidden = self.rnn(intermediate, hidden)

            # Go through out layer
            output = self.out(rnn_out)

            # Set the value of outputs to the correct value
            outputs[:, step, :] = output

        return outputs


class SSGANPerceptronFactory(NetworkFactory):

    @property
    def gen_input_size(self):
        return 128

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = SSGeneratorNet(
            self.loss_function,
            self.num_classes,
            Sequential(
                nn.Linear(128, 256),
                nn.Dropout(0.1),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.Dropout(0.1),
                nn.LeakyReLU(0.2),
                nn.Linear(512, self.input_data_size),
                nn.Tanh()), self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net

    def create_discriminator(self, parameters=None, encoded_parameters=None):
        net = SSDiscriminatorNet(
            self.loss_function,
            self.num_classes,
            Sequential(
                nn.Linear(self.input_data_size, 256),
                nn.Dropout(0.1),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2)
            ),
            Sequential(nn.Linear(512, self.num_classes + 1)),
            self.gen_input_size,
        )

        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net


class SSGANConvolutionalNetworkFactory(NetworkFactory):

    complexity = 128

    @property
    def gen_input_size(self):
        return 100, 1, 1

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = SSGeneratorNet(
            self.loss_function,
            self.num_classes,
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
        net = SSDiscriminatorNet(
            self.loss_function,
            self.num_classes,
            Sequential(
                nn.Conv2d(3, self.complexity, 4, 2, 1),
                nn.Dropout2d(0.1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity, self.complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 2),
                nn.Dropout2d(0.1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 2, self.complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 4),
                nn.Dropout2d(0.1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 4, self.complexity * 8, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 8),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            Sequential(nn.Conv2d(self.complexity * 8, self.num_classes + 1, 4, 1, 0)),
            self.gen_input_size,
            mnist_28x28_conv=False
        )

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

class SSGANConvolutionalMNISTNetworkFactory(NetworkFactory):

    complexity = 128

    @property
    def gen_input_size(self):
        return 100, 1, 1

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = SSGeneratorNet(
            self.loss_function,
            self.num_classes,
            nn.Sequential(
                nn.ConvTranspose2d(100, self.complexity * 8, 4, 1, 0),
                nn.BatchNorm2d(self.complexity * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.complexity * 8, self.complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.complexity * 4, self.complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.complexity * 2, self.complexity, 4, 2, 1),
                nn.BatchNorm2d(self.complexity),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.complexity, 1, 4, 2, 1),
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
        net = SSDiscriminatorNet(
            self.loss_function,
            self.num_classes,
            Sequential(
                nn.Conv2d(1, self.complexity, 4, 2, 1),
                nn.Dropout2d(0.2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity, self.complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 2),
                nn.Dropout2d(0.2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 2, self.complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 4),
                nn.Dropout2d(0.2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 4, self.complexity * 8, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 8),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            Sequential(nn.Conv2d(self.complexity * 8, self.num_classes + 1, 4, 1, 0)),
            self.gen_input_size,
            mnist_28x28_conv=False
        )

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

class SSGANConvMNIST28x28NetworkFactory(NetworkFactory):

    # complexity = 64
    complexity = 128

    @property
    def gen_input_size(self):
        return 100, 1, 1

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = SSGeneratorNet(
            self.loss_function,
            self.num_classes,
            nn.Sequential(
                nn.ConvTranspose2d(100, self.complexity * 4, 4, 1, 0),
                nn.BatchNorm2d(self.complexity * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.complexity * 4, self.complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.complexity * 2, self.complexity, 4, 2, 1),
                nn.BatchNorm2d(self.complexity),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(self.complexity, 1, 4, 2, 3),
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
        net = SSDiscriminatorNet(
            self.loss_function,
            self.num_classes,
            Sequential(
                nn.Conv2d(1, self.complexity, 4, 2, 3),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity, self.complexity * 2, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.complexity * 2, self.complexity * 4, 4, 2, 1),
                nn.BatchNorm2d(self.complexity * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            Sequential(nn.Conv2d(self.complexity * 4, self.num_classes + 1, 4, 1, 0)),
            self.gen_input_size,
            mnist_28x28_conv=True
        )

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