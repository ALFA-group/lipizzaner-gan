import logging
from abc import ABC, abstractmethod

from helpers.coev_losswise_logger import CoevLosswiseLogger
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.population import Population
from training.nn_trainer import NeuralNetworkTrainer


class NaturalEvolutionStrategyTrainer(NeuralNetworkTrainer, ABC):
    _logger = logging.getLogger(__name__)

    @abstractmethod
    def train(self, n_iterations, stop_event=None):
        pass

    def __init__(self, dataloader, network_factory, sigma=0.1, alpha=0.001, population_size=50):
        """
        :param sigma: Noise standard deviation. Read from config if set there.
        :param alpha: Learning rate. Read from config if set there.
        :param population_size: Number of search points created per iteration. Read from config if set there.
        """
        super().__init__(dataloader, network_factory)

        settings = ConfigurationContainer.instance().settings['trainer']['params']

        self._sigma = settings.get('sigma', sigma)
        self._alpha = settings.get('alpha', alpha)
        self._population_size = settings.get('population_size', population_size)

        self.lw_cache = CoevLosswiseLogger(self.__class__.__name__)

    def initialize_populations(self):
        gen = self.network_factory.create_generator()
        dis = self.network_factory.create_discriminator()

        population_gen = Population(individuals=[Individual(genome=gen, fitness=gen.default_fitness)],
                                    default_fitness=gen.default_fitness)

        population_dis = Population(individuals=[Individual(genome=dis, fitness=dis.default_fitness)],
                                    default_fitness=dis.default_fitness)

        return population_gen, population_dis
