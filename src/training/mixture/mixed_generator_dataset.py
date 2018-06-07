import collections
import numpy as np
import torch.utils.data

from helpers.pytorch_helpers import noise


class MixedGeneratorDataset(torch.utils.data.Dataset):

    def __init__(self, generator_population, weights, n_samples):
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
        self.gen_indices = np.random.choice(len(self.individuals), n_samples, p=list(weights.values()))

        self.z = noise(n_samples, self.individuals[0].genome.data_size)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.individuals[self.gen_indices[index]].genome.net(self.z[index].unsqueeze(0)).data.squeeze()
