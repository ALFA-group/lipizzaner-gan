import math
import collections
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch
from helpers.pytorch_helpers import noise
from helpers.pytorch_helpers import to_pytorch_variable

cuda = True if torch.cuda.is_available() else False


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
        if z is None:
            self.z = noise(n_samples, self.individuals[0].genome.data_size)
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
        #label_emb = nn.Embedding(10, 10)
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor 
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        labels = LongTensor(np.random.randint(0, 10, 1)) #random labels for generator input
        #print(self.z.shape)
        #label_emb = self.individuals[self.gen_indices[index]].genome.label_emb
        #return self.individuals[self.gen_indices[index]].genome.net(torch.cat((label_emb(label), self.z[index].unsqueeze(0)), -1)).data.squeeze()
        labels = labels.view(-1,1)
        labels_onehot = FloatTensor(1, 10)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels, 1)
        labels = to_pytorch_variable(labels_onehot.type(FloatTensor))
        return self.individuals[self.gen_indices[index]].genome.net(torch.cat((labels, self.z[index].unsqueeze(0)), -1)).data.squeeze()
