from torch import nn
from torch.nn import Sequential

from networks.network_factory import NetworkFactory
from networks.competetive_net import DiscriminatorNet, GeneratorNet


class MOOCFourLayerPerceptronFactory(NetworkFactory):

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


