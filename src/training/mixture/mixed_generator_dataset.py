import math
import collections
import numpy as np
import torch.utils.data

from helpers.configuration_container import ConfigurationContainer

from helpers.pytorch_helpers import (
    to_pytorch_variable,
    is_cuda_enabled,
    noise,
)



class MixedGeneratorDataset(torch.utils.data.Dataset):

    def __init__(self, generator_population, weights, n_samples, mixture_generator_samples_mode, z=None):
        """
        Creates samples from a mixture of generators, with sample probability defined given a random noise vector
        sampled from the latent space by a weights vector

        :param generator_population: Population of generators that will be used to create the images
        :param weights: Dictionary that maps generator IDs to weights, e.g. {'127.0.0.1:5000': 0.8, '127.0.0.1:5001': 0.2}
        :param n_samples: Number of samples that will be generated
        :param mixture_generator_samples_mode:
        :param z: Noise vector from latent space. If it is not given it generates a new one
        """
        self.n_samples = n_samples
        self.individuals = sorted(generator_population.individuals, key=lambda x: x.source)
        for individual in self.individuals:
            individual.genome.net.eval()
        self.data = []

        self.cc = ConfigurationContainer.instance()

        weights = collections.OrderedDict(sorted(weights.items()))
        weights = {k: v for k, v in weights.items() if any([i for i in self.individuals if i.source == k])}
        weights_np = np.asarray(list(weights.values()))

        if np.sum(weights_np) != 1:
            weights_np = weights_np / np.sum(weights_np).astype(float)    # A bit of patching, but normalize it again

        if mixture_generator_samples_mode == 'independent_probability':
            self.gen_indices = np.random.choice(len(self.individuals), n_samples, p=weights_np.tolist())
        elif mixture_generator_samples_mode == 'exact_proportion':
            # Does not perform checking here if weights_np.tolist() sum up to one
            # There will be some trivial error if prob*n_samples is not integer for prob in weights_np.tolist()
            self.gen_indices = [
                i for gen_idx, prob in enumerate(weights_np.tolist()) for i in [gen_idx] * math.ceil(n_samples * prob)
            ]
            np.random.shuffle(self.gen_indices)
            self.gen_indices = self.gen_indices[:n_samples]
        else:
            raise NotImplementedError(
                "Invalid argument for mixture_generator_samples_mode: {}".format(mixture_generator_samples_mode)
            )

        num_classes = self.individuals[0].genome.num_classes if hasattr(self.individuals[0].genome, 'num_classes') \
                                                                and self.individuals[0].genome.num_classes != 0 else 0

        if z is None:
            z = noise(n_samples, self.individuals[0].genome.data_size)

            if num_classes != 0 and self.cc.settings["network"]["name"] == 'conditional_four_layer_perceptron':
                FloatTensor = torch.cuda.FloatTensor if is_cuda_enabled() else torch.FloatTensor
                LongTensor = torch.cuda.LongTensor if is_cuda_enabled() else torch.LongTensor
                self.labels = LongTensor(np.random.randint(0, num_classes, n_samples))  # random labels between 0 and 9, output of shape batch_size

                self.labels = self.labels.view(-1, 1)
                labels_onehot = torch.FloatTensor(n_samples, num_classes)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, self.labels, 1)

                input_labels = to_pytorch_variable(labels_onehot.type(FloatTensor))

                self.z = torch.cat((input_labels, z), -1)
            else:
                self.z = z

        else:
            self.z = z

        #HACK: If it's a sequential model, add another dimension to the noise input
        # Also we're currently just using a fixed sequence length for sequence generation; make this
        # able to be specified by the user.
        if self.individuals[0].genome.name in ["DiscriminatorSequential", "GeneratorSequential"]:
            sequence_length = 100
            self.z = self.z.unsqueeze(1).repeat(1,sequence_length,1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.individuals[self.gen_indices[index]].genome.net(self.z[index].unsqueeze(0)).data.squeeze()
