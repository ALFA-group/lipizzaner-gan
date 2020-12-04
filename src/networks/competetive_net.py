from abc import abstractmethod, ABC

import copy
import logging
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

        self._logger = logging.getLogger(__name__)

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
    def compute_loss_against(self, opponent, input):
        """
        :return: (computed_loss, output_data (optional))
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
    def __init__(
            self, loss_function, net, data_size, optimize_bias=True,
    ):
        CompetetiveNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = 0

    @property
    def name(self):
        return 'Generator'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input):
        batch_size = input.size(0)

        real_labels = to_pytorch_variable(torch.ones(batch_size))

        z = noise(batch_size, self.data_size)

        fake_images = self.net(z)
        outputs = opponent.net(fake_images).view(-1)

        return self.loss_function(outputs, real_labels), fake_images



class DiscriminatorNet(CompetetiveNet):
    def __init__(
            self, loss_function, net, data_size, optimize_bias=True, disc_output_reshape=None,
    ):
        CompetetiveNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.disc_output_reshape = disc_output_reshape
        self.num_classes = 0

    @property
    def name(self):
        return 'Discriminator'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input):

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

        return d_loss_real + d_loss_fake, None


class GeneratorNetSequential(CompetetiveNet):
    @property
    def name(self):
        return 'GeneratorSequential'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input):
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

        return self.loss_function(outputs, real_labels), fake_sequences


class DiscriminatorNetSequential(CompetetiveNet):
    @property
    def name(self):
        return 'DiscriminatorSequential'

    @property
    def default_fitness(self):
        return float('-inf')

    def compute_loss_against(self, opponent, input):
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

        return d_loss_real + d_loss_fake, None


class ConditionalGeneratorNet(CompetetiveNet):
    def __init__(
            self, loss_function, net, num_classes, data_size, optimize_bias=True,
    ):
        GeneratorNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes
    # pass in loss_function as torch.nn.MSELoss()

    def clone(self):
        return ConditionalGeneratorNet(
            self.loss_function,
            copy.deepcopy(self.net),
            self.num_classes,
            self.data_size,
            self.optimize_bias,
        )

    @property
    def name(self):
        return 'ConditionalGenerator'

    @property
    def default_fitness(self):
        return float('-inf')
    #
    # def generate_samples_and_labels(self, size=10, z=None, labels=None):
    #     FloatTensor = torch.cuda.FloatTensor if is_cuda_enabled() else torch.FloatTensor
    #     LongTensor = torch.cuda.LongTensor if is_cuda_enabled() else torch.LongTensor
    #     if z is None:
    #         z = noise(size, self.data_size)
    #
    #     if labels is None:
    #         labels = LongTensor(np.random.randint(0, self.num_classes, size))  # random labels between 0 and 9, output of shape batch_size
    #
    #     labels = labels.view(-1, 1)
    #     labels_onehot = torch.FloatTensor(size, self.num_classes)
    #     labels_onehot.zero_()
    #     labels_onehot.scatter_(1, labels, 1)
    #
    #     input_labels = to_pytorch_variable(labels_onehot.type(FloatTensor))
    #
    #     gen_input = torch.cat((input_labels, z), -1)
    #
    #     fake_images = self.net(gen_input)
    #
    #     # labels = FloatTensor(labels, size)
    #     import pandas as  pd
    #     print(fake_images.detach().numpy().shape)
    #     print(labels.detach().numpy().shape)
    #     xx = np.concatenate((fake_images.detach().numpy(), labels.detach().numpy()), axis=1) # torch.cat((fake_images, labels), -1)
    #     df = pd.DataFrame(xx)
    #     df.to_csv('/home/jamal/test.csv')
    #
    #     return fake_images, labels





    def compute_loss_against(self, opponent, input, labels = None, alpha = None, beta = None, iter = None, log_class_distribution = False,):
        FloatTensor = torch.cuda.FloatTensor if is_cuda_enabled() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if is_cuda_enabled() else torch.LongTensor
        batch_size = input.size(0)
        # print(batch_size)
        # print(input.size(1))
        # print('batch size')
        # print(batch_size)
        real_labels = to_pytorch_variable(torch.ones(batch_size))  # label all generator images 1 (real)

        z = noise(batch_size, self.data_size)  # dims: batch size x data_size

        labels = LongTensor(
            np.random.randint(0, self.num_classes, batch_size))  # random labels between 0 and 9, output of shape batch_size
        labels = labels.view(-1, 1)
        labels_onehot = torch.FloatTensor(batch_size, self.num_classes)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels, 1)
        # print(labels_onehot)


        labels = to_pytorch_variable(labels_onehot.type(FloatTensor))
        # print(labels)
        # print(self.label_emb(labels))
        # concatenate z and labels here before passing into generator net

        gen_input = torch.cat((labels, z), -1)
        # print(gen_input)
        # print('gen input shape')
        # print(gen_input.shape)
        fake_images = self.net(gen_input)
        # print('fake images shape')
        # print(fake_images.shape)
        # fake_images = fake_images.view(fake_images.size(0), *)

        dis_input = torch.cat((fake_images, labels), -1)  # discriminator training data input
        # concatenate fake_images and labels here before passing into discriminator net
        outputs = opponent.net(dis_input).view(-1)  # view(-1) flattens tensor

        return self.loss_function(outputs,
                                  real_labels), fake_images  # loss function evaluated discriminator output vs. 1 (generator trying to get discriminator output to be 1)


# discriminator has to take in class labels (conditioned variable) and images

class ConditionalDiscriminatorNet(CompetetiveNet):

    def __init__(self, loss_function, net, num_classes, data_size, optimize_bias=True):
        DiscriminatorNet.__init__(self, loss_function, net, data_size,optimize_bias=optimize_bias)
        self.num_classes = num_classes

    @property
    def name(self):
        return 'ConditionalDiscriminator'

    def clone(self):
        return ConditionalDiscriminatorNet(
            self.loss_function,
            copy.deepcopy(self.net),
            self.num_classes,
            self.data_size,
            self.optimize_bias,
        )

    @property
    def default_fitness(self):
        return float('-inf')



    def compute_loss_against(self, opponent, input, labels = None, alpha = None, beta = None, iter = None, log_class_distribution = False,):

        # need to pass in the labels from dataloader too in lipizzaner_gan_trainer.py
        # Compute loss using real images
        # Second term of the loss is always zero since real_labels == 1
        batch_size = input.size(0)

        FloatTensor = torch.cuda.FloatTensor if is_cuda_enabled() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if is_cuda_enabled() else torch.LongTensor

        real_labels = torch.Tensor(batch_size)
        real_labels.fill_(0.9)
        real_labels = to_pytorch_variable(real_labels)

        fake_labels = to_pytorch_variable(torch.zeros(batch_size))

        labels = labels.view(-1, 1).cuda() if is_cuda_enabled() else labels.view(-1, 1)
        labels_onehot = torch.FloatTensor(batch_size, self.num_classes)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels, 1)

        labels = to_pytorch_variable(labels_onehot.type(FloatTensor))

        instance_noise_std_dev_min = 0.5
        instance_noise_std_dev_max = 5.0
        instance_noise_std_dev = 2.5
        instance_noise_mean = 0


        # Adding instance noise to prevent Discriminator from getting too strong
        if iter is not None:
            std = max(
                instance_noise_std_dev_min,
                instance_noise_std_dev_max - iter * 0.001,
            )
        else:
            instance_noise_std_dev

        input_perturbation = to_pytorch_variable(
            torch.empty(input.shape).normal_(mean=instance_noise_mean, std=std)
        )

        input = input + input_perturbation

        dis_input = torch.cat((input, labels), -1)  # discriminator training data input

        outputs = self.net(dis_input).view(-1)  # pass in training data input and respective labels to discriminator
        d_loss_real = self.loss_function(outputs, real_labels)  # get real image loss of discriminator (output vs. 1)

        # torch.cat((img.view(img.size(0), -1), self.label_embedding(gen_labels)), -1)

        # Compute loss using fake images
        # First term of the loss is always zero since fake_labels == 0
        gen_labels = LongTensor(np.random.randint(0, self.num_classes, batch_size))  # random labels for generator input

        z = noise(batch_size, self.data_size)  # noise for generator input

        gen_labels = gen_labels.view(-1, 1)
        labels_onehot = torch.FloatTensor(batch_size, self.num_classes)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, gen_labels, 1)

        gen_labels = to_pytorch_variable(labels_onehot.type(FloatTensor))

        gen_input = torch.cat((gen_labels, z), -1)

        fake_images = opponent.net(gen_input)
        # print('fake images shape')
        # print(fake_images.shape)
        dis_input = torch.cat((fake_images, gen_labels), -1)  # discriminator training data input
        outputs = self.net(dis_input).view(-1)
        d_loss_fake = self.loss_function(outputs, fake_labels)  # get fake image loss of discriminator (output vs. 0)

        return (d_loss_real + d_loss_fake), None