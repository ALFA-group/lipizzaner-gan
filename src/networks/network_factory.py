from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import RNN
from torch.autograd import Variable

from helpers.configuration_container import ConfigurationContainer
from networks.competetive_net import DiscriminatorNet, GeneratorNet, GeneratorNetSequential, DiscriminatorNetSequential, GeneratorNetConditional, DiscriminatorNetConditional, GeneratorNetConditional_2, DiscriminatorNetConditional_2


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

class ConditionalNetworkFactory (NetworkFactory):

    @property
    def gen_input_size(self):
        return 64 #dimensionality of latent space (noise input) 

    def create_generator(self, parameters=None, encoded_parameters=None):

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        net = GeneratorNetConditional(
            self.loss_function,
            Sequential(
                *block(self.gen_input_size + 10, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, self.input_data_size),
                nn.Tanh()), self.gen_input_size
            )


        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net

    def create_discriminator(self, parameters=None, encoded_parameters=None):

        net = DiscriminatorNetConditional(
            self.loss_function,
            nn.Sequential(
            nn.Linear(self.input_data_size + 10, 512), #MNIST images are 28 x 28 x 1 pixels, so 784 input neurons, CIFAR images are 3 x 32 x 32 = 3072 pixels
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        ), self.gen_input_size)




        if parameters is not None:
            net.parameters = parameters
        if encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters

        return net        

class ConditionalNetworkFactory_2 (NetworkFactory):

    @property
    def gen_input_size(self):
        return 64 #dimensionality of latent space (noise input) 

    def create_generator(self, parameters=None, encoded_parameters=None):

        net = GeneratorNetConditional(
            self.loss_function,
            Sequential(
                nn.Linear(self.gen_input_size + 10, 256),
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

        net = DiscriminatorNetConditional(
            self.loss_function,
            Sequential(
                nn.Linear(self.input_data_size + 10, 256),
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

class GeneratorCust(nn.Module):

        def __init__(self):
            super(GeneratorCust, self).__init__()
            self.complexity = 64

            self.main = nn.Sequential(
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
                nn.Tanh())

        def forward(self, z):
            output = self.main(z)

            return output

class DiscriminatorCust(nn.Module):
    def __init__(self):
        super(DiscriminatorCust, self).__init__()
            
        self.complexity = 64

        self.main = nn.Sequential(
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
                nn.Sigmoid())


    def forward(self, x):
        output = self.main(x)
        
        return output.view(-1, 1).squeeze(1) 

class ConvolutionalNetworkFactory(NetworkFactory):

    #complexity = 64

    @property
    def gen_input_size(self):
        return 100, 1, 1

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = GeneratorNet(
            self.loss_function,
            GeneratorCust(),
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
            DiscriminatorCust(),
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

class GeneratorCustom(nn.Module):

        def __init__(self):
            super(GeneratorCustom, self).__init__()
            self.complexity = 64
            self.ylabel=nn.Sequential(
                nn.Linear(10,1000),
                nn.ReLU(True)
            )
            
            self.yz=nn.Sequential(
                nn.Linear(100,200),
                nn.ReLU(True)
            )

            self.main = nn.Sequential(
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
                )
#add dropout in the layers

        def forward(self, z,y):
        
            #mapping noise and labe
            z=self.yz(z)
            y=self.ylabel(y)
            
            #mapping concatenated input to the main generator network
            inp=torch.cat([z,y],1)
            inp=inp.view(-1,1200,1,1)
            output = self.main(inp)

            return output

class DiscriminatorCustom(nn.Module):
        def __init__(self):
            super(DiscriminatorCustom, self).__init__()
                
            self.complexity = 64

            self.ylabel=nn.Sequential(
                nn.Linear(10,64*64*1),
                nn.ReLU(True)
            )

            self.main = nn.Sequential(
                nn.Conv2d(4, self.complexity, 4, 2, 1),
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
                )


        def forward(self, x,y):
            y=self.ylabel(y)
            y=y.view(-1,1,64,64)
            inp=torch.cat([x,y],1)
            output = self.main(inp)
            
            return output.view(-1, 1).squeeze(1)

class ConvolutionalNetworkFactoryConditional(NetworkFactory):

    '''def __init__(self, input_data_size, loss_function=None):
        super(ConvolutionalNetworkFactoryConditional, self).__init__(input_data_size, loss_function)
        self.generator = GeneratorCustom()
        self.discriminator = DiscriminatorCustom()'''
            

    @property
    def gen_input_size(self):
        return 100

    def create_generator(self, parameters=None, encoded_parameters=None):
        net = GeneratorNetConditional_2(
            self.loss_function,
            GeneratorCustom(),
            self.gen_input_size)

        if parameters is not None:
            net.parameters = parameters
        elif encoded_parameters is not None:
            net.encoded_parameters = encoded_parameters
        else:
            net.net.apply(self._init_weights)

        return net

    def create_discriminator(self, parameters=None, encoded_parameters=None):
        net = DiscriminatorNetConditional_2(
            self.loss_function,
            DiscriminatorCustom(),
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
