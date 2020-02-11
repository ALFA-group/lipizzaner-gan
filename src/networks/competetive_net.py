from abc import abstractmethod, ABC

import copy

import numpy as np
import torch
from torch.nn import Softmax

from distribution.state_encoder import StateEncoder
from helpers.pytorch_helpers import to_pytorch_variable, is_cuda_enabled, size_splits, noise

class CompetetiveNet(ABC):
    def __init__(self, loss_function, net, data_size, optimize_bias=True):
        self.data_size = data_size
        self.net = net.cuda() if is_cuda_enabled() else net
        self.optimize_bias = optimize_bias

        self.loss_function = loss_function
        if self.loss_function.__class__.__name__ == 'MustangsLoss':
            self.loss_function.set_network_name(self.name)

        try:
            self.n_weights = np.sum([l.weight.numel() for l in self.net if hasattr(l, 'weight')])
            # Calculate split positions; cumulative sum needed because split() expects positions, not chunk sizes
            self.split_positions_weights = [l.weight.numel() for l in self.net if hasattr(l, 'weight')]

            if optimize_bias:
                self.split_positions_biases = [l.bias.numel() for l in self.net if hasattr(l, 'bias')]
        except Exception as e:
            print(e)

    @abstractmethod
    def compute_loss_against(self, opponent, input, labels=None):
        """
        :return: (computed_loss, output_data -> (optional), accuracy(s) -> (optional))
        """
        pass

    def clone(self):
        return eval(self.__class__.__name__)(self.loss_function, copy.deepcopy(self.net),
                                             self.data_size, self.optimize_bias)

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def default_fitness(self):
        pass

    @property
    def encoded_parameters(self):
        """
        :return: base64 encoded representation of the networks state dictionary
        """
        return StateEncoder.encode(self.net.state_dict())

    @encoded_parameters.setter
    def encoded_parameters(self, value):
        """
        :param value: base64 encoded representation of the networks state dictionary
        """
        self.net.load_state_dict(StateEncoder.decode(value))

    @property
    def parameters(self):
        """
        :return: 1d-ndarray[nr_of_layers * (nr_of_weights_per_layer + nr_of_biases_per_layer)]
        """
        weights = torch.cat([l.weight.data.view(l.weight.numel()) for l in self.net if hasattr(l, 'weight')])
        if self.optimize_bias:
            biases = torch.cat([l.bias.data for l in self.net if hasattr(l, 'bias')])
            return torch.cat((weights, biases))
        else:
            return weights

    @parameters.setter
    def parameters(self, value):
        """
        :param value: 1d-ndarray[nr_of_layers * (nr_of_weights_per_layer + nr_of_biases_per_layer)]
        """

        if self.optimize_bias:
            (weights, biases) = value.split(self.n_weights)
        else:
            weights = value

        # Update weights
        layered_weights = size_splits(weights, self.split_positions_weights)
        for i, layer in enumerate([l for l in self.net if hasattr(l, 'weight')]):
            self._update_layer_field(layered_weights[i], layer.weight)

        # Update biases
        if self.optimize_bias:
            layered_biases = size_splits(biases, self.split_positions_biases)
            for i, layer in enumerate([l for l in self.net if hasattr(l, 'bias')]):
                self._update_layer_field(layered_biases[i], layer.bias)

    @staticmethod
    def _update_layer_field(source, target):
        # Required because it's recommended to only use in-place operations on PyTorch variables
        target.data.zero_()
        if len(target.data.shape) == 1:
            target.data.add_(source)
        else:
            target.data.add_(source.view(target.size()))


class GeneratorNet(CompetetiveNet):
    @property
    def name(self):
        return 'Generator'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input, labels=None):
        batch_size = input.size(0)

        real_labels = to_pytorch_variable(torch.ones(batch_size))

        z = noise(batch_size, self.data_size)

        fake_images = self.net(z)
        outputs = opponent.net(fake_images).view(-1)

        return self.loss_function(outputs, real_labels), fake_images, None



class DiscriminatorNet(CompetetiveNet):
    @property
    def name(self):
        return 'Discriminator'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input, labels=None):

        # If HeuristicLoss is applied in the Generator, the Discriminator applies BCELoss
        if self.loss_function.__class__.__name__ == 'MustangsLoss':
            if 'HeuristicLoss' in self.loss_function.get_applied_loss_name():
                self.loss_function.set_applied_loss(torch.nn.BCELoss())

        # Compute loss using real images
        # Second term of the loss is always zero since real_labels == 1
        batch_size = input.size(0)

        real_labels = to_pytorch_variable(torch.ones(batch_size))
        fake_labels = to_pytorch_variable(torch.zeros(batch_size))

        outputs = self.net(input).view(-1)
        d_loss_real = self.loss_function(outputs, real_labels)

        # Compute loss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = noise(batch_size, self.data_size)
        fake_images = opponent.net(z)
        outputs = self.net(fake_images).view(-1)
        d_loss_fake = self.loss_function(outputs, fake_labels)

        return d_loss_real + d_loss_fake, None, None


class GeneratorNetSequential(CompetetiveNet):
    @property
    def name(self):
        return 'GeneratorSequential'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input, labels=None):
        batch_size = input.size(0)
        sequence_length = input.size(1)
        num_inputs = input.size(2)
        # batch_size = input.shape[0]

        # Define differently based on whether we're evaluating entire sequences as true or false, vs. individual messages.
        real_labels = to_pytorch_variable(torch.ones(batch_size))

        z = noise(batch_size, self.data_size)

        # Repeats the noise to match the shape
        new_z = z.unsqueeze(1).repeat(1,sequence_length,1)
        fake_sequences = self.net(new_z)

        outputs_intermediate = opponent.net(fake_sequences)

        # Compute BCELoss using D(G(z))
        sm = Softmax()
        outputs = sm(outputs_intermediate[:, -1, :].contiguous().view(-1))

        return self.loss_function(outputs, real_labels), fake_sequences, None


class DiscriminatorNetSequential(CompetetiveNet):
    @property
    def name(self):
        return 'DiscriminatorSequential'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input, labels=None):
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1

        batch_size = input.size(0)
        sequence_length = input.size(1)
        num_inputs = input.size(2)

        real_labels = to_pytorch_variable(torch.ones(batch_size))
        fake_labels = to_pytorch_variable(torch.zeros(batch_size))

        outputs_intermediate = self.net(input)
        sm = Softmax()

        outputs = sm(outputs_intermediate[:, -1, :].contiguous().view(-1))
        d_loss_real = self.loss_function(outputs, real_labels)

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = noise(batch_size, self.data_size)
        new_z = z.unsqueeze(1).repeat(1,sequence_length,1)

        fake_images = opponent.net(new_z)
        outputs_full = self.net(fake_images)
        sm = Softmax()

        outputs = sm(outputs_full[:, -1, :].contiguous().view(-1))
        d_loss_fake = self.loss_function(outputs, fake_labels)

        return d_loss_real + d_loss_fake, None, None


class SSDiscriminatorNet(DiscriminatorNet):

    def __init__(self, adv_loss, label_pred_loss, num_classes, net, data_size,
                 adv_layer, label_pred_layer, optimize_bias=True, conv=False):
        DiscriminatorNet.__init__(self, adv_loss, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes
        self.adv_layer = adv_layer
        self.label_pred_layer = label_pred_layer
        self.label_pred_loss = label_pred_loss
        self.conv = conv

    @property
    def name(self):
        return 'SemiSupervisedDiscriminator'

    @property
    def default_fitness(self):
        return float('-inf')

    def clone(self):
        return SSDiscriminatorNet(self.loss_function,
                                  self.label_pred_loss,
                                  self.num_classes,
                                  copy.deepcopy(self.net),
                                  self.data_size,
                                  self.adv_layer,
                                  self.label_pred_layer,
                                  self.optimize_bias)

    def compute_loss_against(self, opponent, input, labels=None):
        """
        Assumes input is provided as a list and that the last column is an
        integer column representing the class of the corresponding input
        """
        # Compute CrossEntropyLoss using real images
        batch_size = input.size(0)
        cuda = is_cuda_enabled()
        tensor = torch.Tensor(batch_size) if cuda else torch.Tensor(batch_size).cuda()
        tensor.fill_(self.num_classes)
        tensor = tensor.long()
        fake_labels = to_pytorch_variable(tensor)

        ones = torch.ones(batch_size) if cuda else torch.ones(batch_size).cuda()
        real = to_pytorch_variable(ones)
        zeros = torch.zeros(batch_size) if cuda else torch.zeros(batch_size).cuda()
        fake = to_pytorch_variable(zeros)

        # Real Loss
        network_output = self.net(input)
        if not self.conv:
            network_output = network_output.view(network_output.shape[0], -1)

        label_prediction = self.label_pred_layer(network_output).view(-1, self.num_classes + 1)
        label_prediction_loss = self.label_pred_loss(label_prediction, labels)
        outputs = self.adv_layer(network_output).view(-1)
        validity = self.loss_function(outputs, real)
        real_acc = np.mean(outputs.data.cpu().numpy())
        d_loss_real = (label_prediction_loss + validity) / 2

        pred = label_prediction.data.cpu().numpy()
        gt = labels.data.cpu().numpy()
        pred_labels = np.argmax(pred, axis=1)
        d_acc = np.mean(pred_labels == gt)

        # Fake Loss
        z = noise(batch_size, self.data_size)
        fake_images = opponent.net(z)
        network_output = self.net(fake_images)
        if not self.conv:
            network_output = network_output.view(network_output.shape[0], -1)

        label_prediction = self.label_pred_layer(network_output).view(-1, self.num_classes + 1)
        label_prediction_loss = self.label_pred_loss(label_prediction, fake_labels)
        outputs = self.adv_layer(network_output).view(-1)
        validity = self.loss_function(outputs, fake)
        fake_acc = np.mean(outputs.data.cpu().numpy())
        d_loss_fake = (label_prediction_loss + validity) / 2

        return d_loss_real + d_loss_fake, None, (d_acc, real_acc, fake_acc)


class SSGeneratorNet(GeneratorNet):

    def __init__(self, loss_function, num_classes, net, data_size, optimize_bias=True):
        GeneratorNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes

    @property
    def name(self):
        return 'SemiSupervisedGenerator'

    def clone(self):
        return SSGeneratorNet(self.loss_function,
                                  self.num_classes,
                                  copy.deepcopy(self.net),
                                  self.data_size,
                                  self.optimize_bias)

    def compute_loss_against(self, opponent: SSDiscriminatorNet, input, labels=None):
        batch_size = input.size(0)

        ones = torch.ones(batch_size) if is_cuda_enabled() else torch.ones(batch_size)
        real_labels = to_pytorch_variable(ones)

        z = noise(batch_size, self.data_size)

        fake_images = self.net(z)
        outputs = opponent.adv_layer(opponent.net(fake_images)).view(-1)

        # Compute BCELoss using D(G(z))
        return self.loss_function(outputs, real_labels), fake_images, None
