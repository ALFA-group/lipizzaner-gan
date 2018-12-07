import math
import collections
import numpy as np
import torch.utils.data

from helpers.pytorch_helpers import noise


class MixedGeneratorDataset(torch.utils.data.Dataset):

    def __init__(self, generator_population, weights, n_samples, mixture_generator_samples_mode):
        """
        Creates samples from a mixture of generators, with sample probability defined by a weights vector

        :param generator_population: Population of generators that will be used to create the images
        :param weights: Dictionary that maps generator IDs to weights, e.g. {'127.0.0.1:5000': 0.8, '127.0.0.1:5001': 0.2}
        :param n_samples: Number of samples that will be generated
        """
        self.n_samples = n_samples
        self.individuals = sorted(generator_population.individuals, key=lambda x: x.source)
        for individual in self.individuals:
            individual.genome.net.eval()
        self.data = []

        weights = collections.OrderedDict(sorted(weights.items()))
        weights = {k: v for k, v in weights.items() if any([i for i in self.individuals if i.source == k])}
        weights_np = np.asarray(list(weights.values()))
        if np.sum(weights_np) != 1:
            weights_np = weights_np / np.sum(weights_np).astype(float)    # Abit of patching, but normalize it again

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

        self.z = noise(n_samples, self.individuals[0].genome.data_size)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.individuals[self.gen_indices[index]].genome.net(self.z[index].unsqueeze(0)).data.squeeze()
