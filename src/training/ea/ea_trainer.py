import logging
import random
from abc import ABC, abstractmethod

import numpy as np
import torch

from helpers.coev_losswise_logger import CoevLosswiseLogger
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.population import Population, TYPE_GENERATOR, TYPE_DISCRIMINATOR
from training.nn_trainer import NeuralNetworkTrainer


class EvolutionaryAlgorithmTrainer(NeuralNetworkTrainer, ABC):
    _logger = logging.getLogger(__name__)

    @abstractmethod
    def train(self, n_iterations, stop_event=None):
        pass

    def __init__(self, dataloader, network_factory, population_size=10, tournament_size=2, mutation_probability=0.9,
                 n_replacements=1, sigma=0.25, alpha=0.25):
        """
        :param population_size: Number of elements per population. Read from config if set there.
        :param tournament_size: Number of elements per tournament selection. Read from config if set there.
        :param mutation_probability: Probability of each weight's mutation. Read from config if set there.
        :param n_replacements: Number of population elements replaced by new ones. Read from config if set there.
        :param sigma: Noise standard deviation. Read from config if set there.
        :param alpha: Learning rate. Read from config if set there.
        """

        self.settings = ConfigurationContainer.instance().settings['trainer']['params']

        self._alpha = self.settings.get('alpha', alpha)
        self._sigma = self.settings.get('sigma', sigma)
        self._n_replacements = self.settings.get('n_replacements', n_replacements)
        self._mutation_probability = self.settings.get('mutation_probability', mutation_probability)
        self._population_size = self.settings.get('population_size', population_size)
        self._tournament_size = self.settings.get('tournament_size', tournament_size)

        super().__init__(dataloader, network_factory)

        self.lw_cache = CoevLosswiseLogger(self.__class__.__name__)

    def mutate_gaussian(self, population):
        """
        Mutation is done by creating a gaussian distributed delta array and applying it to the original parameters array
        """

        # Calculate graussian distributed param mutations
        individual_len = len(population.individuals[0].genome.parameters)
        deltas = np.random.normal(loc=0, scale=self._sigma, size=(len(population.individuals), individual_len))

        # Set delta values to 0 with p(1 - mutation_probability).
        # Done with numpy because PyTorch's boolean indexing is broken.
        deltas[np.random.rand(*deltas.shape) < 1 - self._mutation_probability] = 0

        for i, individual in enumerate(population.individuals):
            params = individual.genome.parameters

            params += torch.from_numpy(deltas[i]).float() * self._alpha
            individual.genome.parameters = params

    def tournament_selection(self, population, population_type, is_logging=False):
        assert 0 < self._tournament_size <= len(population.individuals), \
            "Invalid tournament size: {}".format(self._tournament_size)

        competition_population = Population(individuals=[], default_fitness=population.default_fitness)
        new_population = Population(individuals=[], default_fitness=population.default_fitness,
                                    population_type=population_type)

        # Iterate until there are enough tournament winners selected
        while len(new_population.individuals) < self._population_size:
            # Randomly select tournament size individual solutions
            # from the population.
            competitors = random.sample(population.individuals, self._tournament_size)
            competition_population.individuals = competitors

            # Rank the selected solutions
            competition_population.sort_population()

            # Copy the solution
            winner = competitors[0].clone()
            winner.is_local = True
            winner.fitness = competition_population.default_fitness

            # Append the best solution to the winners
            new_population.individuals.append(winner)

        if is_logging:
            new_individuals_names = [individual.name for individual in new_population.individuals]
            self._logger.info('{} are selected from tournament selection'.format(new_individuals_names))

        return new_population

    def initialize_populations(self):
        populations = [None] * 2
        populations[TYPE_GENERATOR] = Population(individuals=[], default_fitness=0, population_type=TYPE_GENERATOR)
        populations[TYPE_DISCRIMINATOR] = Population(individuals=[], default_fitness=0,
                                                     population_type=TYPE_DISCRIMINATOR)

        for i in range(self._population_size):
            gen, dis = self.network_factory.create_both()
            populations[TYPE_GENERATOR].individuals.append(Individual(genome=gen, fitness=gen.default_fitness))
            populations[TYPE_DISCRIMINATOR].individuals.append(Individual(genome=dis, fitness=dis.default_fitness))

        populations[TYPE_GENERATOR].default_fitness = populations[TYPE_GENERATOR].individuals[0].fitness
        populations[TYPE_DISCRIMINATOR].default_fitness = populations[TYPE_DISCRIMINATOR].individuals[0].fitness

        return populations[TYPE_GENERATOR], populations[TYPE_DISCRIMINATOR]

    @staticmethod
    def evaluate_fitness_against_population(population_gen, population_dis, input_data):
        for i in range(len(population_dis.individuals)):
            for j in range(len(population_gen.individuals)):
                # Fitness values need to be calculated for both generator and discriminator,
                # because the respective methods may differ depending on the problem.
                fitness_disc = float(population_dis.individuals[i].genome.compute_loss_against(
                    population_gen.individuals[j].genome, input_data)[0])

                if fitness_disc > population_dis.individuals[i].fitness:
                    population_dis.individuals[i].fitness = fitness_disc

                fitness_gen = float(population_gen.individuals[j].genome.compute_loss_against(
                    population_dis.individuals[i].genome, input_data)[0])

                if fitness_gen > population_gen.individuals[j].fitness:
                    population_gen.individuals[j].fitness = fitness_gen
