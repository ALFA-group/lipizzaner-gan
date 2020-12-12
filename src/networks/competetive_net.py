import copy
import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from distribution.state_encoder import StateEncoder
from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import (is_cuda_enabled, noise, size_splits,
                                     to_pytorch_variable)
from torch.nn import BCELoss, CrossEntropyLoss, Softmax
from torch.nn.utils.weight_norm import WeightNorm


class CompetetiveNet(ABC):
    def __init__(self, loss_function, net, data_size, optimize_bias=True):
        self.data_size = data_size
        self.net = net.cuda() if is_cuda_enabled() else net
        self.optimize_bias = optimize_bias
        self._logger = logging.getLogger(__name__)
        self.loss_function = loss_function
        if self.loss_function.__class__.__name__ == "MustangsLoss":
            self.loss_function.set_network_name(self.name)

        try:
            self.n_weights = np.sum([layer.weight.numel() for layer in self.net if hasattr(layer, "weight")])
            # Calculate split positions; cumulative sum needed because split() expects positions, not chunk sizes
            self.split_positions_weights = [layer.weight.numel() for layer in self.net if hasattr(layer, "weight")]

            if optimize_bias:
                self.split_positions_biases = [layer.bias.numel() for layer in self.net if hasattr(layer, "bias")]
        except Exception as e:
            print(e)

    @abstractmethod
    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
    ):
        """
        :return: (computed_loss, output_data -> (optional), accuracy(s) -> (optional))
        """
        pass

    def clone(self):
        return eval(self.__class__.__name__)(
            self.loss_function,
            copy.deepcopy(self.net),
            self.data_size,
            self.optimize_bias,
        )

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
        weights = torch.cat(
            [layer.weight.data.view(layer.weight.numel()) for layer in self.net if hasattr(layer, "weight")]
        )
        if self.optimize_bias:
            biases = torch.cat([layer.bias.data for layer in self.net if hasattr(layer, "bias")])
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
        for i, layer in enumerate([layerr for layerr in self.net if hasattr(layerr, "weight")]):
            self._update_layer_field(layered_weights[i], layer.weight)

        # Update biases
        if self.optimize_bias:
            layered_biases = size_splits(biases, self.split_positions_biases)
            for i, layer in enumerate([layerr for layerr in self.net if hasattr(layerr, "bias")]):
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
        self,
        loss_function,
        net,
        data_size,
        optimize_bias=True,
    ):
        CompetetiveNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = 0

    @property
    def name(self):
        return "Generator"

    @property
    def default_fitness(self):
        return float("-inf")

    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
        diverse_fitness=False,
    ):
        batch_size = input.size(0)


        z = noise(batch_size, self.data_size)
        real_labels = to_pytorch_variable(torch.ones(batch_size))

        if diverse_fitness:
            fake_labels = to_pytorch_variable(torch.zeros(batch_size))
            with torch.no_grad():
                fake_images = self.net(z)

            real_outputs = opponent.net(input).view(-1)
            fake_outputs = opponent.net(fake_images).view(-1)
            output_loss = self.loss_function(real_outputs, real_labels) + self.loss_function(fake_outputs, fake_labels)

            gradients = torch.autograd.grad(outputs=output_loss, inputs=opponent.net.parameters(),
                                        grad_outputs=torch.ones(output_loss.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
            with torch.no_grad():
                allgrad = gradients[0]
                for grad in gradients[1:]:
                    grad = grad.view(-1)
                    allgrad = torch.cat([allgrad,grad])

            Fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()
            Fq = self.loss_function(fake_outputs, real_labels)

            return alpha*Fd + beta*Fq, fake_images, None

        else:
            fake_images = self.net(z)
            outputs = opponent.net(fake_images).view(-1)
            return self.loss_function(outputs, real_labels), fake_images, None


class DiscriminatorNet(CompetetiveNet):
    def __init__(
        self,
        loss_function,
        net,
        data_size,
        optimize_bias=True,
        disc_output_reshape=None,
    ):
        CompetetiveNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.disc_output_reshape = disc_output_reshape
        self.num_classes = 0

    @property
    def name(self):
        return "Discriminator"

    @property
    def default_fitness(self):
        return float("-inf")

    def clone(self):
        return DiscriminatorNet(
            self.loss_function,
            copy.deepcopy(self.net),
            self.data_size,
            self.optimize_bias,
            disc_output_reshape=self.disc_output_reshape,
        )

    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
        diverse_fitness=False,
    ):

        # If HeuristicLoss is applied in the Generator, the Discriminator applies BCELoss
        if self.loss_function.__class__.__name__ == "MustangsLoss":
            if "HeuristicLoss" in self.loss_function.get_applied_loss_name():
                self.loss_function.set_applied_loss(torch.nn.BCELoss())

        # Compute loss using real images
        # Second term of the loss is always zero since real_labels == 1
        batch_size = input.size(0)

        if self.disc_output_reshape is not None:
            input = input.view(self.disc_output_reshape)

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
        return "GeneratorSequential"

    @property
    def default_fitness(self):
        return float("-inf")

    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
    ):
        batch_size = input.size(0)
        sequence_length = input.size(1)

        # Define differently based on whether we're evaluating entire sequences as true or false, vs. individual messages.
        real_labels = to_pytorch_variable(torch.ones(batch_size))

        z = noise(batch_size, self.data_size)

        # Repeats the noise to match the shape
        new_z = z.unsqueeze(1).repeat(1, sequence_length, 1)
        fake_sequences = self.net(new_z)

        outputs_intermediate = opponent.net(fake_sequences)

        # Compute BCELoss using D(G(z))
        sm = Softmax()
        outputs = sm(outputs_intermediate[:, -1, :].contiguous().view(-1))

        return self.loss_function(outputs, real_labels), fake_sequences, None


class DiscriminatorNetSequential(CompetetiveNet):
    @property
    def name(self):
        return "DiscriminatorSequential"

    @property
    def default_fitness(self):
        return float("-inf")

    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
    ):
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1

        batch_size = input.size(0)
        sequence_length = input.size(1)

        real_labels = to_pytorch_variable(torch.ones(batch_size))
        fake_labels = to_pytorch_variable(torch.zeros(batch_size))

        outputs_intermediate = self.net(input)
        sm = Softmax()

        outputs = sm(outputs_intermediate[:, -1, :].contiguous().view(-1))
        d_loss_real = self.loss_function(outputs, real_labels)

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = noise(batch_size, self.data_size)
        new_z = z.unsqueeze(1).repeat(1, sequence_length, 1)

        fake_images = opponent.net(new_z)
        outputs_full = self.net(fake_images)
        sm = Softmax()

        outputs = sm(outputs_full[:, -1, :].contiguous().view(-1))
        d_loss_fake = self.loss_function(outputs, fake_labels)

        return d_loss_real + d_loss_fake, None, None


class SSDiscriminatorNet(DiscriminatorNet):
    def __init__(
        self,
        label_pred_loss,
        num_classes,
        net,
        classification_layer,
        data_size,
        optimize_bias=True,
        disc_output_reshape=None,
    ):
        DiscriminatorNet.__init__(self, label_pred_loss, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes
        self.classification_layer = classification_layer.cuda() if is_cuda_enabled() else classification_layer
        self.disc_output_reshape = disc_output_reshape

        cc = ConfigurationContainer.instance()
        self.instance_noise_mean = cc.settings["network"].get("in_mean", 0.0)
        self.instance_noise_std_dev = cc.settings["network"].get("in_std", 1e-10)
        self.instance_noise_std_decay_rate = cc.settings["network"].get("in_std_decay_rate", 0.0)
        self.instance_noise_std_dev_min = cc.settings["network"].get("in_std_min", 1e-10)
        self.instance_noise_fake_image_decay = cc.settings["network"].get("in_fake_decay", False)
        self.label_rate = cc.settings["dataloader"].get("label_rate", 1)

    @property
    def name(self):
        return "SemiSupervisedDiscriminator"

    @property
    def default_fitness(self):
        return float("-inf")

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
        return SSDiscriminatorNet(
            self.loss_function,
            self.num_classes,
            copy.deepcopy(self.net),
            copy.deepcopy(self.classification_layer),
            self.data_size,
            self.optimize_bias,
            disc_output_reshape=self.disc_output_reshape,
        )

    def _get_labeled_mask(self, batch_size):
        label_mask = to_pytorch_variable(torch.zeros(batch_size))
        label_count = to_pytorch_variable(torch.tensor(batch_size * self.label_rate).int())
        label_mask[range(label_count)] = 1.0
        return label_mask

    def _log_classification_distribution(self, ground_truth, label_mask, labels, num_usable_labels, pred_labels):
        frequencies = labels.bincount(minlength=self.num_classes).float()
        frequencies[frequencies == 0.0] = 1e-10
        frequencies = frequencies.data.cpu().numpy()
        usable_labels = label_mask * labels.float()

        label_data_per_class = usable_labels.long().bincount(minlength=self.num_classes)
        label_data_per_class[0] = num_usable_labels - label_data_per_class[1:].sum()
        predicted_frequencies = np.bincount(
            pred_labels[np.where(pred_labels == ground_truth)],
            minlength=self.num_classes,
        )
        classification_rate_per_class = torch.from_numpy(predicted_frequencies * 100.0 / frequencies)

        labels = to_pytorch_variable(torch.tensor([i for i in range(0, self.num_classes)]))
        statistics = zip(labels, label_data_per_class, classification_rate_per_class)

        stats = f"\nLabel, Number of Labeled Data points, Classification Rate for this label"
        for (label, label_data, classification_rate) in statistics:
            stats += f"\n{label}, {label_data}, {classification_rate}"
        self._logger.info(stats)

    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
    ):
        batch_size = input.size(0)
        tensor = torch.Tensor(batch_size)
        tensor.fill_(self.num_classes)
        tensor = tensor.long()
        fake_unsupervised_labels = to_pytorch_variable(tensor)

        label_mask = self._get_labeled_mask(batch_size)

        # Positive Label Smoothing at 0.9 following the Improved Techniques for
        # Training GANs. This prevents the discriminator from getting
        # overconfident with regards to its predictions and makes it less
        # susceptible to adversarial examples
        real_unsupervised_labels = torch.Tensor(batch_size)
        real_unsupervised_labels.fill_(0.9)
        real_unsupervised_labels = to_pytorch_variable(real_unsupervised_labels)

        # Adding instance noise to prevent Discriminator from getting too strong
        if iter is not None:
            std = max(
                self.instance_noise_std_dev_min,
                self.instance_noise_std_dev - iter * self.instance_noise_std_decay_rate,
            )
        else:
            std = self.instance_noise_std_dev
        input_perturbation = to_pytorch_variable(
            torch.empty(input.shape).normal_(mean=self.instance_noise_mean, std=std)
        )
        input = input + input_perturbation

        if self.disc_output_reshape is not None:
            input = input.view(self.disc_output_reshape)

        network_output = self.classification_layer(self.net(input))
        network_output = network_output.view(batch_size, -1)

        # Real Supervised Loss
        supervised_loss_function = CrossEntropyLoss(reduction="none")
        supervised_loss = supervised_loss_function(network_output, labels)
        num_usable_labels = torch.sum(label_mask)
        _logger = logging.getLogger(__name__)
        _logger.info(f"Num Usable Labels: {num_usable_labels}")
        loss_for_usable_labels = torch.sum(supervised_loss * label_mask)
        label_prediction_loss = loss_for_usable_labels / num_usable_labels

        # Real Unsupervised Loss
        softmax_layer = Softmax()
        probabilities = softmax_layer(network_output)
        real_probabilities = -probabilities[:, -1] + 1
        bce_loss = BCELoss()
        validity = bce_loss(real_probabilities, real_unsupervised_labels)

        d_loss_supervised = label_prediction_loss
        d_loss_unsupervised = validity

        pred = network_output.data.cpu().numpy()
        ground_truth = labels.data.cpu().numpy()
        predicted_labels = np.argmax(pred, axis=1)
        accuracy = np.mean(predicted_labels == ground_truth)

        if log_class_distribution:
            self._log_classification_distribution(
                ground_truth,
                label_mask,
                labels,
                num_usable_labels.long(),
                predicted_labels,
            )

        # Fake Unsupervised Loss
        z = noise(batch_size, self.data_size)
        fake_images = opponent.net(z)

        if self.instance_noise_fake_image_decay and iter is not None:
            std = max(
                self.instance_noise_std_dev_min,
                self.instance_noise_std_dev - iter * self.instance_noise_std_decay_rate,
            )
        else:
            std = self.instance_noise_std_dev
        fake_image_perturbation = to_pytorch_variable(
            torch.empty(fake_images.shape).normal_(mean=self.instance_noise_mean, std=std)
        )
        fake_images = fake_images + fake_image_perturbation

        network_output = self.classification_layer(self.net(fake_images))
        network_output = network_output.view(batch_size, -1)
        label_prediction_loss = self.loss_function(network_output, fake_unsupervised_labels)
        d_loss_unsupervised = d_loss_unsupervised + label_prediction_loss

        if alpha is not None:
            loss = alpha * d_loss_unsupervised + beta * d_loss_supervised
        else:
            loss = d_loss_unsupervised + d_loss_supervised
        return loss, None, accuracy


class SSGeneratorNet(GeneratorNet):
    def __init__(
        self,
        loss_function,
        num_classes,
        net,
        data_size,
        optimize_bias=True,
        use_feature_matching=False,
    ):
        GeneratorNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes
        self.use_feature_matching = use_feature_matching

    @property
    def name(self):
        return "SemiSupervisedGenerator"

    def clone(self):
        return SSGeneratorNet(
            self.loss_function,
            self.num_classes,
            copy.deepcopy(self.net),
            self.data_size,
            self.optimize_bias,
            use_feature_matching=self.use_feature_matching,
        )

    def compute_loss_against(
        self,
        opponent: SSDiscriminatorNet,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
    ):
        batch_size = input.size(0)

        z = noise(batch_size, self.data_size)
        fake_images = self.net(z)
        network_output = opponent.net(fake_images)

        if not self.use_feature_matching:
            fake = to_pytorch_variable(torch.zeros(batch_size))
            network_output = opponent.classification_layer(network_output)
            network_output = network_output.view(batch_size, -1)

            softmax_layer = Softmax()
            probabilities = softmax_layer(network_output)
            fake_probabilities = probabilities[:, -1]
            bce_loss = BCELoss()
            loss = bce_loss(fake_probabilities, fake)
        else:
            # Feature Matching
            if opponent.disc_output_reshape is not None:
                input = input.view(opponent.disc_output_reshape)
            real_data_moments = torch.mean(opponent.net(input), 0)
            fake_data_moments = torch.mean(network_output, 0)

            loss = torch.mean(torch.abs(real_data_moments - fake_data_moments))

        return loss, fake_images, None


class ConditionalGeneratorNet(CompetetiveNet):
    def __init__(
        self,
        loss_function,
        net,
        num_classes,
        data_size,
        optimize_bias=True,
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
        return "ConditionalGenerator"

    @property
    def default_fitness(self):
        return float("-inf")

    def generate_samples_and_labels(self, size=10, z=None, labels=None):
        FloatTensor = torch.cuda.FloatTensor if is_cuda_enabled() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if is_cuda_enabled() else torch.LongTensor
        if z is None:
            z = noise(size, self.data_size)

        if labels is None:
            labels = LongTensor(
                np.random.randint(0, self.num_classes, size)
            )  # random labels between 0 and 9, output of shape batch_size

        labels = labels.view(-1, 1)
        labels_onehot = torch.FloatTensor(size, self.num_classes)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels, 1)

        input_labels = to_pytorch_variable(labels_onehot.type(FloatTensor))

        gen_input = torch.cat((input_labels, z), -1)

        fake_images = self.net(gen_input)

        return fake_images, labels

    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
    ):
        FloatTensor = torch.cuda.FloatTensor if is_cuda_enabled() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if is_cuda_enabled() else torch.LongTensor
        batch_size = input.size(0)
        real_labels = to_pytorch_variable(torch.ones(batch_size))  # label all generator images 1 (real)

        z = noise(batch_size, self.data_size)  # dims: batch size x data_size

        labels = LongTensor(
            np.random.randint(0, self.num_classes, batch_size)
        )  # random labels between 0 and 9, output of shape batch_size
        labels = labels.view(-1, 1)
        labels_onehot = torch.FloatTensor(batch_size, self.num_classes)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels, 1)

        labels = to_pytorch_variable(labels_onehot.type(FloatTensor))

        gen_input = torch.cat((labels, z), -1)
        fake_images = self.net(gen_input)

        dis_input = torch.cat((fake_images, labels), -1)  # discriminator training data input
        # concatenate fake_images and labels here before passing into discriminator net
        outputs = opponent.net(dis_input).view(-1)  # view(-1) flattens tensor

        return (
            self.loss_function(outputs, real_labels),
            fake_images,
        )  # loss function evaluated discriminator output vs. 1 (generator trying to get discriminator output to be 1)


class ConditionalDiscriminatorNet(CompetetiveNet):
    # discriminator has to take in class labels (conditioned variable) and images
    def __init__(self, loss_function, net, num_classes, data_size, optimize_bias=True):
        DiscriminatorNet.__init__(self, loss_function, net, data_size, optimize_bias=optimize_bias)
        self.num_classes = num_classes

    @property
    def name(self):
        return "ConditionalDiscriminator"

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
        return float("-inf")

    def compute_loss_against(
        self,
        opponent,
        input,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
        log_class_distribution=False,
    ):
        # need to pass in the labels from dataloader too in lipizzaner_gan_trainer.py
        # Compute loss using real images
        # Second term of the loss is always zero since real_labels == 1
        batch_size = input.size(0)

        FloatTensor = torch.cuda.FloatTensor if is_cuda_enabled() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if is_cuda_enabled() else torch.LongTensor
        real_labels = to_pytorch_variable(torch.ones(batch_size))
        fake_labels = to_pytorch_variable(torch.zeros(batch_size))

        labels = labels.view(-1, 1).cuda() if is_cuda_enabled() else labels.view(-1, 1)
        labels_onehot = torch.FloatTensor(batch_size, self.num_classes)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels, 1)

        labels = to_pytorch_variable(labels_onehot.type(FloatTensor))

        dis_input = torch.cat((input, labels), -1)  # discriminator training data input

        outputs = self.net(dis_input).view(-1)  # pass in training data input and respective labels to discriminator
        d_loss_real = self.loss_function(outputs, real_labels)  # get real image loss of discriminator (output vs. 1)

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
        dis_input = torch.cat((fake_images, gen_labels), -1)  # discriminator training data input
        outputs = self.net(dis_input).view(-1)
        d_loss_fake = self.loss_function(outputs, fake_labels)  # get fake image loss of discriminator (output vs. 0)

        return (d_loss_real + d_loss_fake), None
