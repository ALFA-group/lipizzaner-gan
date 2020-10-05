import os
import copy

import torch

from helpers.pytorch_helpers import is_cuda_enabled


class Individual:
    def __init__(
        self,
        genome,
        fitness,
        is_local=True,
        learning_rate=None,
        optimizer_state=None,
        source=None,
        id=None,
        iteration=0,
    ):
        """
        :param genome: A neural network, i.e. a subclass of CompetitveNet (Discriminator or Generator)
        """
        self.genome = genome
        self.fitness = fitness
        self.is_local = is_local
        self.learning_rate = learning_rate
        self.optimizer_state = optimizer_state
        self.source = source
        self.id = id

        # To keep track of which iteration the current individual is in (for logging and tracing purpose)
        self.iteration = iteration

    @staticmethod
    def decode(
        create_genome,
        params,
        fitness_tensor=None,
        is_local=True,
        learning_rate=None,
        optimizer_state=None,
        source=None,
        id=None,
        iteration=None,
    ):
        """
        Creates a new instance from encoded parameters and a fitness tensor
        :param params: 1d-Tensor containing all the weights for the individual
        :param fitness_tensor: 0d-Tensor containing exactly one fitness value
        :param create_genome: Function that creates either a generator or a discriminator network
        :return:
        """
        genome = create_genome(encoded_parameters=params)
        fitness = float(fitness_tensor) if fitness_tensor is not None else float("-inf")

        return Individual(
            genome,
            fitness,
            is_local,
            learning_rate,
            optimizer_state,
            source,
            id,
            iteration,
        )

    def clone(self):
        return Individual(
            self.genome.clone(),
            self.fitness,
            self.is_local,
            self.learning_rate,
            copy.deepcopy(self.optimizer_state),
            self.source,
            self.id,
            self.iteration,
        )

    @property
    def name(self):
        """ Uniquely identify an individual (for further logging purpose) """
        return "Iteration{}:{}:{}".format(self.iteration, self.source, self.id)

    def save_genome(self, network_prefix, output_dir, include_classification_layer=False):
        """ Saves the network defined by the genome """
        self.source.replace(":", "-")
        filename = "{}{}.pkl".format(network_prefix, self.source)
        torch.save(
            self.genome.net.state_dict(),
            os.path.join(output_dir, filename),
        )

        if include_classification_layer:
            torch.save(
                self.genome.classification_layer.state_dict(),
                os.path.join(output_dir, "{}{}_classification_layer.pkl".format(network_prefix, self.source)),
            )
        return filename
