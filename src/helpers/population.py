import logging
import itertools

from helpers.pytorch_helpers import calculate_net_weights_dist


TYPE_GENERATOR = 0
TYPE_DISCRIMINATOR = 1

ALWAYS_REPLACE = False


class Population:
    _logger = logging.getLogger(__name__)

    def __init__(self, individuals, default_fitness, population_type=None):
        self.individuals = individuals
        self.default_fitness = default_fitness
        self.population_type = population_type

    def sort_population(self, is_logging=False):
        self.individuals.sort(key=lambda x: x.fitness)

        if is_logging:
            sorted_individuals_names = [individual.name for individual in self.individuals]
            self._logger.info("Current local sorted population is {}".format(sorted_individuals_names))

    def replacement(self, new_population, n_replacements=1, is_logging=False):
        new_population.sort_population()
        self.sort_population()

        replacer_individuals_names = []  # Individuals who replace others
        replacee_individuals_names = []  # Individuals who are replaced by others

        # TODO break out early
        for i in range(n_replacements):
            j = i - n_replacements
            if ALWAYS_REPLACE or self.individuals[j].fitness > new_population.individuals[i].fitness:
                replacer_individuals_names.append(new_population.individuals[i].name)
                replacee_individuals_names.append(self.individuals[j].name)

                self.individuals[j] = new_population.individuals[i].clone()
                self.individuals[j].is_local = True

        if is_logging:
            self._logger.info("Replacers are {}".format(replacer_individuals_names))
            self._logger.info("Replacees are {}".format(replacee_individuals_names))

    @property
    def net_weights_dist(self):
        net_weights_dist = []
        for ind1, ind2 in itertools.combinations(self.individuals, 2):
            # Each tuple is of format (ind1, ind2, value)
            net_weights_dist.append(
                (ind1.name, ind2.name, calculate_net_weights_dist(ind1.genome.net, ind2.genome.net),)
            )

        return net_weights_dist
