"""
Author: Jamal Toutouh (toutouh@mit.edu) - www.jamal.es

This code is part of the research of our paper "Re-purposing Heterogeneous Generative Ensembles with Evolutionary
Computation" presented during GECCO 2020 (https://doi.org/10.1145/3377930.3390229)

evolutionary_restricted_ensemble_optimization.py contains the code RestrictedEnsembleOptimization class, which
implements REO-GEN optimization problem defined to create ensembles with a specific size.
"""
import numpy as np
import glob
import sys

import math
import random

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class GAGeneratorIndividual(list):
    def __init__(self, attributes):
        # Some initialisation with received values
        self.attr1 = attributes
        pass


class RestrictedEnsembleOptimization:
    def __init__(self, ensemble_size, generators_prefix='mnist-generator', generators_path='./mnist-generators/'):
        precision = 100
        self.precision = precision
        self.possible_weights = self.create_weights_list(0, self.precision+1, 1)
        self.generators_path = generators_path
        self.generators_prefix = generators_prefix
        self.generators_in_path = self.get_maximum_generators_index(generators_path, generators_prefix)
        self.ensemble_size = ensemble_size

    def mutate(self, individual):
        """It calls the mutation operator to be applied to an individual.
        :return: A tuple of one individual.
        """
        prob = random.random()
        if prob < 0.333:
            mutation_type = 'just-weight-probability'
        elif prob < 0.666:
            mutation_type = 'just-generator-index'
        else:
            mutation_type = 'generator-and-weight'
        individual, = self.mutations(individual, mutation_type=mutation_type)
        return individual,

    def mutations(self, individual, mutation_type='generator-and-weight'):
        """Mutate an individual by replacing attributes.
        :param individual: :term:`Sequence <sequence>` individual to be mutated.
        :param mutation_type: Type of mutation to be applied (i.e., 'just-weight-probability', 'just-generator-index',
                              and 'generator-and-weight'
        :return: A tuple of one individual.
        """
        decimals_precision = 2
        size = len(individual)
        probability = 1/size
        if mutation_type == 'just-weight-probability':
            for i in range(size):
                if random.random() < probability:
                    genome_js = list(range(size))
                    random.shuffle(genome_js)
                    for genome_j in genome_js:
                        if i != genome_j:
                            break
                    weight_j, generator_index_j = math.modf(individual[genome_j])
                    weight, generator_index = math.modf(individual[i])
                    sum_weights = round(weight + weight_j, decimals_precision)
                    weight = round(random.uniform(0, sum_weights), decimals_precision)
                    weight_j = sum_weights - weight
                    individual[i] = round(generator_index + weight, decimals_precision)
                    individual[genome_j] = round(generator_index_j + weight_j, decimals_precision)
        elif mutation_type == 'just-generator-index':
            generators_individual = [int(gen) for gen in individual]
            possible_generators_index = set(list(range(self.generators_in_path))).difference(set(generators_individual))
            for i in range(size):
                if random.random() < probability:
                    weight, generator_index = math.modf(individual[i])
                    possible_generators_index.add(int(individual[i]))
                    new_generator = random.sample(possible_generators_index, 1)[0]
                    possible_generators_index.remove(new_generator)
                    individual[i] = round(new_generator + weight, decimals_precision)
        elif mutation_type == 'generator-and-weight':
            generators_individual = [int(gen) for gen in individual]
            possible_generators_index = set(list(range(self.generators_in_path))).difference(set(generators_individual))
            for i in range(size):
                if random.random() < probability:
                    genome_js = list(range(size))
                    random.shuffle(genome_js)
                    for genome_j in genome_js:
                        if i != genome_j:
                            break
                    weight_j, generator_index_j = math.modf(individual[genome_j])
                    weight, generator_index = math.modf(individual[i])
                    sum_weights = round(weight + weight_j, decimals_precision)
                    weight = round(random.uniform(0, sum_weights), decimals_precision)
                    weight_j = sum_weights - weight
                    possible_generators_index.add(int(individual[i]))
                    new_generator = random.sample(possible_generators_index, 1)[0]
                    possible_generators_index.remove(new_generator)
                    individual[i] = round(new_generator + weight, decimals_precision)
                    individual[genome_j] = round(generator_index_j + weight_j, decimals_precision)
        return individual,

    def find_repeated(self, value, individual, search_range):
        """Mutate an individual by replacing attributes.
        :param value: Value to find.
        :param individual: :term:`Sequence <sequence>` individual to be mutated.
        :param search_range: Search positions to find the repeated value.
        :return: Index of the repeated value or -1.
        """
        for i in search_range:
            if value == individual[i]:
                return i
        return -1

    def correct_solution(self, offspring, parent, cxpoint1, cxpoint2):
        """Applies the correction process after crossover.
        :param offspring: The offspring individual.
        :param parent: The parent individual.
        :param cxpoint1: The cxpoint1 of crossover operator.
        :param cxpoint2: The cxpoint2 of crossover operator.
        :return: The offspring after correction if needed.
        """
        generators_offspring = [int(gen) for gen in offspring]
        generators_parent = [int(gen) for gen in parent]
        possible_generators_index = set(generators_parent).difference(set(generators_offspring))
        search_range = list(range(0, cxpoint1)) + list(range(cxpoint2, len(offspring)))
        for i in range(cxpoint1, cxpoint2):
            index = self.find_repeated(generators_offspring[i], generators_offspring, search_range)
            if index >= 0:
                if self.find_repeated(generators_parent[i], generators_offspring, list(
                        range(len(offspring)))) < 0:  # The possible replacement is in the solution already
                    offspring[index] = parent[i]
                    generators_offspring[index] = generators_parent[i]
                    possible_generators_index = possible_generators_index.difference({generators_offspring[index]})
                else:
                    generators_offspring[index] = possible_generators_index.pop()
                    offspring[index] = generators_offspring[index] + math.modf(offspring[index])[0]
        return offspring

    def cxTwoPointGAN(self, ind1, ind2, method='average-weights'):
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param method: Crossover method to be applied
        :return: A tuple of two individuals.
        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        decimals_precision = 2
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        parent_ind1 = ind1.copy()
        parent_ind2 = ind2.copy()
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        ind1 = self.correct_solution(ind1, parent_ind1, cxpoint1, cxpoint2)

        ind2 = self.correct_solution(ind2, parent_ind2, cxpoint1, cxpoint2)

        if method == 'preserve-weights':
            for i in range(cxpoint1, cxpoint2):
                weight_1, generators_index1 = math.modf(ind1[i])
                weight_2, generators_index2 = math.modf(ind2[i])
                ind1[i] = generators_index1 + round(weight_2, decimals_precision)
                ind2[i] = generators_index2 + round(weight_1, decimals_precision)
        elif method == 'average-weights':
            first = True
            for i in range(len(ind1)):
                weight_1, generators_index1 = math.modf(ind1[i])
                weight_2, generators_index2 = math.modf(ind2[i])
                sum_weights = round(weight_1 + weight_2, decimals_precision)
                digit_control = int(round(self.precision * sum_weights))
                if digit_control % 2 == 1 and first:
                    weight_1 = round((sum_weights - 1/self.precision) / 2 + 1/self.precision, decimals_precision)
                    weight_2 = round(sum_weights - weight_1, decimals_precision)
                    first = False
                elif digit_control % 2 == 1 and not first:
                    weight_2 = round((sum_weights - 1/self.precision) / 2 + 1/self.precision, decimals_precision)
                    weight_1 = round(sum_weights - weight_2, decimals_precision)
                    first = True
                else:
                    weight_1 = round(sum_weights / 2, decimals_precision)
                    weight_2 = round(sum_weights - weight_1, decimals_precision)
                ind1[i] = generators_index1 + round(weight_1, decimals_precision)
                ind2[i] = generators_index2 + round(weight_2, decimals_precision)
        return ind1, ind2

    def create_individual(self, individual_class):
        """It randomly creates individuals/solutions given the ensemble size and the maximum number of generators
        used to create the ensembles.
        :param individual_class: Class that defines the solutions.
        :return: An individual/solution.
        """
        weights = self.get_weights_tentative(self.ensemble_size)[0]
        generators_indexes =  list(range(self.generators_in_path))
        random.shuffle(generators_indexes)
        generators = [generators_indexes.pop() for _ in range(self.ensemble_size)]
        return individual_class([gen + w for gen, w in zip(generators, weights)])

    def get_all_generators_in_path(self):
        """It gets the maximum number of generators to be used to create the ensembles.
        :return: The maximum number of generators to be used to create the ensembles.
        """
        return len([gen for gen in glob.glob('{}*gene*.pkl'.format(self.generators_path))])

    def get_maximum_generators_index(self, generators_path, generators_prefix):
        """It gets the maximum number of generators to be used to create the ensembles.
        :parameter generators_path: Path where the generator files are stored.
        :parameter generators_prefix: Prefix of the file names that stores the generator model,
        :return: The maximum number of generators to be used to create the ensembles.
        """
        generators_found = len([gen for gen in glob.glob('{}*gene*.pkl'.format(generators_path))])
        i = 0
        print('{}/{}-{:03d}.pkl'.format(generators_path, generators_prefix, i))
        while len([gen for gen in glob.glob('{}/{}-{:03d}.pkl'.format(generators_path, generators_prefix, i))]) > 0:
            i += 1
        if i == 0:
            print('Error: No generators found in the path {} with the prefix {}'.format(generators_path, generators_prefix))
            sys.exit(0)
        if generators_found != i:
            print('Warning! {} found in the path {}, but the algorithm will use just {}. Check the genrators prefix.'
                  .format(generators_found, generators_path, i))
        return i

    def create_weights_list(self, min_value, max_value, step_size):
        """It creates a list of the possible values for the weights to be used to construct an ensemble.
        :param min_value: Min value that a weight can take.
        :param max_value: Max value that a weight can take.
        :param step_size: Step size from min to max value.
        :return: List of the possible values for the weights to be used to construct an ensemble.
        """
        return np.arange(min_value, max_value, step_size)

    def get_generator_for_ensemble(self, generator_index):
        """It returns the full path of a given generator given its index.
        :param generator_index: Generator index.
        :return: Full path of a given generator given its index."""
        return '{}/{}-{:03d}.pkl'.format(self.generators_path, self.generators_prefix, int(generator_index))\
            , '{:03d}'.format(generator_index)

    def get_mixture_from_individual(self, individual):
        """It returns the full path of a set of generators given the list of tuples that defines a solution/ensemble.
        :param weight_and_generator_indices: List that defines a solution/ensemble.
        :return: A tuple that contains the list of the full path of the generators and the indexes of the generators."""
        weight_and_generator_indices = [math.modf(gen) for gen in individual]
        tentative_weights = [weight for weight, generator_index in weight_and_generator_indices]
        return tentative_weights,\
               ['{}/{}-{:03d}.pkl'.format(self.generators_path, self.generators_prefix, int(generator_index))
                for weight, generator_index in weight_and_generator_indices], \
               ['{:03d}'.format(int(generator_index)) for weight, generator_index in weight_and_generator_indices]

    def get_generators_for_ensemble(self, weight_and_generator_indices):
        """It returns the full path of a set of generators given the list of tuples that defines a solution/ensemble.
        :param weight_and_generator_indices: List that defines a solution/ensemble.
        :return: A tuple that contains the list of the full path of the generators and the indexes of the generators."""
        return ['{}/{}-{:03d}.pkl'.format(self.generators_path, self.generators_prefix, int(generator_index))
                for weight, generator_index in weight_and_generator_indices], \
               ['{:03d}'.format(int(generator_index)) for weight, generator_index in weight_and_generator_indices]

    def get_possible_tentative_weights(self, current_prob, i, size):
        """It randomly computes a given weight for the ensemble.
        :param current_prob: Is the sum of the current probabilities in the solution.
        :param i: The index of the weight in the solution.
        :param size: The ensemble size
        :return: A given weight for the ensemble."""
        max_prob = 1 * self.precision
        number_of_following_weights = size - i
        if number_of_following_weights==1:
            return [max_prob - current_prob]
        else:
            max_probability = (max_prob - current_prob) / number_of_following_weights
            return self.possible_weights[self.possible_weights <= max_probability]

    @staticmethod
    def extract_weights(weight_and_generator_indices):
        """It extracts the weights of a solution to be printed.
        :param weight_and_generator_indices: Individual/solution that defines an ensemble.
        :return: List of weights that defines the mixture."""
        return [round(weight,1) for weight, generator_index in weight_and_generator_indices]

    def get_weights_tentative(self, size):
        """It produces a set of weights to define a mixture.
        :param size: Ensemble size.
        :return: List of weights that defines the mixture."""
        result = []
        for i in range(size):
            current_prob = sum(result)
            possible_weights = self.get_possible_tentative_weights(current_prob, i, size)
            result.append(random.choice(possible_weights))
            i += 1
        return np.array(result) / self.precision, len(result)

    def show_mixture(self, individual):
        weight_and_generator_indices = [math.modf(gen) for gen in individual]
        line = 'Size = ' + str(len(individual)) + ' - Generator = ['
        for weight, generator_index in weight_and_generator_indices:
            line += str(int(generator_index)) + '-' + str(round(weight, 2)) + ', '
        return line[:-2] + ']'

    def show_ensemble_size_info(self):
        return 'Ensemble size={}, '.format(self.ensemble_size)

