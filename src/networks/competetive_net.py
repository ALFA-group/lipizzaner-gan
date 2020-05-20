from abc import abstractmethod, ABC

import copy
import logging
import numpy as np
import torch
from torch.nn import Softmax, BCELoss, CrossEntropyLoss

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
    def compute_loss_against(self, opponent, input, labels=None, alpha=None,
                             beta=None, iter=None, get_class_distribution=False):
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

    def compute_loss_against(self, opponent, input, labels=None, alpha=None,
                             beta=None, iter=None, get_class_distribution=False):
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

    def compute_loss_against(self, opponent, input, labels=None, alpha=None,
                             beta=None, iter=None, get_class_distribution=False):

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

    def compute_loss_against(self, opponent, input, labels=None, alpha=None,
                             beta=None, iter=None, get_class_distribution=False):
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

    def compute_loss_against(self, opponent, input, labels=None, alpha=None,
                             beta=None, iter=None, get_class_distribution=False):
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

    def __init__(self, label_pred_loss, num_classes, net, classification_layer, data_size,
                 optimize_bias=True, mnist_28x28_conv=False):
        DiscriminatorNet.__init__(self, label_pred_loss, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes
        self.classification_layer = classification_layer.cuda() if is_cuda_enabled() else classification_layer
        self.mnist_28x28_conv = mnist_28x28_conv

    @property
    def name(self):
        return 'SemiSupervisedDiscriminator'

    @property
    def default_fitness(self):
        return float('-inf')

    @property
    def encoded_classification_layer_parameters(self):
        """
        :return: base64 encoded representation of the classification layer's
        state dictionary
        """
        return StateEncoder.encode(self.classification_layer.state_dict())

    @encoded_classification_layer_parameters.setter
    def encoded_classification_layer_parameters(self, value):
        """
        :param value: base64 encoded representation of the classification
        layer's state dictionary
        """
        self.classification_layer.load_state_dict(StateEncoder.decode(value))

    def clone(self):
        return SSDiscriminatorNet(self.loss_function,
                                  self.num_classes,
                                  copy.deepcopy(self.net),
                                  copy.deepcopy(self.classification_layer),
                                  self.data_size,
                                  self.optimize_bias,
                                  mnist_28x28_conv=self.mnist_28x28_conv)

    def _get_labeled_mask(self, batch_size, label_rate):
        label_mask = to_pytorch_variable(torch.zeros(batch_size))
        label_count = to_pytorch_variable(torch.tensor(batch_size * label_rate).int())
        label_mask[range(label_count)] = 1.0
        return label_mask

    def _log_classification_distribution(self, ground_truth, label_mask, labels,
                                         num_usable_labels, pred_labels):
        frequencies = labels.bincount(minlength=self.num_classes).float()
        frequencies[frequencies == 0.0] = 1e-10
        frequencies = frequencies.data.cpu().numpy()
        usable_labels = label_mask * labels.float()
        label_data_per_class = usable_labels.long().bincount(
            minlength=self.num_classes)
        label_data_per_class[0] = num_usable_labels - label_data_per_class[1:].sum()
        predicted_frequencies = np.bincount(
            pred_labels[np.where(pred_labels == ground_truth)],
            minlength=self.num_classes
        )
        classification_rate_per_class = torch.from_numpy(
            predicted_frequencies * 100.0 / frequencies
        )
        labels = to_pytorch_variable(torch.tensor([i for i in range(0, self.num_classes)]))
        statistics = zip(labels, label_data_per_class, classification_rate_per_class)
        stats = f"\nLabel, Number of Labeled Data points, Classification Rate for this label"
        for (label, label_data, classification_rate) in statistics:
            stats += f"\n{label}, {label_data}, {classification_rate}"
        self._logger.info(stats)

    def compute_loss_against(self, opponent, input, labels=None, alpha=None,
                             beta=None, iter=None, log_class_distribution=False):
        batch_size = input.size(0)
        tensor = torch.Tensor(batch_size)
        tensor.fill_(self.num_classes)
        tensor = tensor.long()
        fake_labels = to_pytorch_variable(tensor)

        label_rate = 0.01
        label_mask = self._get_labeled_mask(batch_size, label_rate)

        # Positive Label Smoothing
        real = torch.Tensor(batch_size)
        real.fill_(0.9)
        real = to_pytorch_variable(real)

        # Adding noise to prevent Discriminator from getting too strong
        if iter is not None:
            std = max(0.065, 0.1 - iter * 0.0005)
            input_perturbation = to_pytorch_variable(torch.empty(input.shape).normal_(mean=0, std=std))
        else:
            input_perturbation = to_pytorch_variable(torch.empty(input.shape).normal_(mean=0, std=0.1))

        # NOTE: Maybe this is not necessary since Dropout might be taking care of this
        # input_perturbation = to_pytorch_variable(torch.empty(input.shape).normal_(mean=0, std=0.1))
        input = input + input_perturbation

        if self.mnist_28x28_conv:
            input = input.view(-1, 1, 28, 28)

        network_output = self.classification_layer(self.net(input))
        network_output = network_output.view(batch_size, -1)

        # Real Supervised Loss
        supervised_loss_function = CrossEntropyLoss(reduction='none')
        supervised_loss = supervised_loss_function(network_output, labels)
        num_usable_labels = torch.sum(label_mask)
        loss_for_usable_labels = torch.sum(supervised_loss * label_mask)
        label_prediction_loss = loss_for_usable_labels / num_usable_labels

        # Real Unsupervised Loss
        softmax_layer = Softmax()
        probabilities = softmax_layer(network_output)
        real_probabilities = -probabilities[:, -1] + 1
        bce_loss = BCELoss()
        validity = bce_loss(real_probabilities, real)

        d_loss_supervised = label_prediction_loss
        d_loss_unsupervised = validity

        pred = network_output.data.cpu().numpy()
        ground_truth = labels.data.cpu().numpy()
        pred_labels = np.argmax(pred, axis=1)
        accuracy = np.mean(pred_labels == ground_truth)

        if log_class_distribution:
            self._log_classification_distribution(
                ground_truth, label_mask, labels, num_usable_labels.long(), pred_labels
            )

        # Fake Unsupervised Loss
        z = noise(batch_size, self.data_size)
        fake_images = opponent.net(z)

        fake_image_perturbation = to_pytorch_variable(torch.empty(fake_images.shape).normal_(mean=0, std=0.1))
        fake_images = fake_images + fake_image_perturbation

        network_output = self.classification_layer(self.net(fake_images))
        network_output = network_output.view(batch_size, -1)
        label_prediction_loss = self.loss_function(network_output, fake_labels)
        d_loss_unsupervised = d_loss_unsupervised + label_prediction_loss

        return d_loss_supervised + d_loss_unsupervised, None, accuracy


class SSGeneratorNet(GeneratorNet):

    def __init__(self, loss_function, num_classes, net, data_size,
                 optimize_bias=True, is_mnist=True):
        GeneratorNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes
        self.is_mnist = is_mnist

    @property
    def name(self):
        return 'SemiSupervisedGenerator'

    def clone(self):
        return SSGeneratorNet(self.loss_function,
                              self.num_classes,
                              copy.deepcopy(self.net),
                              self.data_size,
                              self.optimize_bias,
                              is_mnist=self.is_mnist)

    def compute_loss_against(self, opponent: SSDiscriminatorNet, input,
                             labels=None, alpha=None, beta=None, iter=None,
                             get_class_distribution=False):
        batch_size = input.size(0)

        z = noise(batch_size, self.data_size)
        fake_images = self.net(z)
        network_output = opponent.net(fake_images)

        if self.is_mnist:
            fake = to_pytorch_variable(torch.zeros(batch_size))
            network_output = opponent.classification_layer(network_output)
            network_output = network_output.view(batch_size, -1)

            softmax_layer = Softmax()
            probabilities = softmax_layer(network_output)
            fake_probabilities = probabilities[:, -1]
            bce_loss = BCELoss()
            loss = bce_loss(fake_probabilities, fake)
        else:
            real_data_moments = torch.mean(opponent.net(input), 0)
            fake_data_moments = torch.mean(network_output, 0)

            loss = torch.mean(torch.abs(real_data_moments - fake_data_moments))

        return loss, fake_images, None
