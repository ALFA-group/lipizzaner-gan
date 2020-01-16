import argparse
import logging
import os
import re
import sys

import losswise
import torch
import yaml
import numpy as np
import glob

from helpers.CustomArgumentParser import CustomArgumentParser
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.log_helper import LogHelper
from helpers.population import Population
from helpers.yaml_include_loader import YamlIncludeLoader
from lipizzaner import Lipizzaner
from lipizzaner_client import LipizzanerClient
from lipizzaner_master import LipizzanerMaster, GENERATOR_PREFIX
from networks.network_factory import NetworkFactory
from training.mixture.score_factory import ScoreCalculatorFactory
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset

import time
from tvd_based_constructor import TVDBasedConstructor
from random_search_ensemble_generator import TVDBasedRandomSearch

from deap import base
from deap import creator
from deap import tools
import math
import random
from itertools import repeat
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class GAGeneratorIndividual(list):
    def __init__(self, attributes):
        # Some initialisation with received values
        self.attr1 = attributes
        pass


class TVDBasedGA:
    def __init__(self, precision=100, generators_path='./generators/', mode='iterative'):
        self.precision = precision
        self.possible_weights = self.create_weights_list(0, self.precision+1, 1)
        self.generators_path = generators_path
        self.generators_in_path = self.get_all_generators_in_path()

    def mutate(self, individual, low, up, indpb):
        prob = random.random()
        if prob < 0.333:
            mutation_type = 'just-weight-probabilty'
        elif prob < 0.666:
            mutation_type = 'just-generator-index'
        else:
            mutation_type = 'generator-and-weight'
        individual, = self.mutations(individual, low, up, indpb, mutation_type=mutation_type)
        return individual,

    def mutations(self, individual, low, up, indpb, mutation_type='generator-and-weight'):
        """Mutate an individual by replacing attributes, with probability *indpb*,
        by a integer uniformly drawn between *low* and *up* inclusively.
        :param individual: :term:`Sequence <sequence>` individual to be mutated.
        :param low: The lower bound or a :term:`python:sequence` of
                    of lower bounds of the range from wich to draw the new
                    integer.
        :param up: The upper bound or a :term:`python:sequence` of
                   of upper bounds of the range from wich to draw the new
                   integer.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.
        """
        decimals_precision = 2
        size = len(individual)
        if not isinstance(low, Sequence):
            low = repeat(low, size)
        elif len(low) < size:
            raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
        if not isinstance(up, Sequence):
            up = repeat(up, size)
        elif len(up) < size:
            raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

        if mutation_type == 'just-weight-probabilty':
            for i, xl, xu in zip(range(size), low, up):
                if random.random() < indpb:
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
            for i, xl, xu in zip(range(size), low, up):
                if random.random() < indpb:
                    weight, generator_index = math.modf(individual[i])
                    individual[i] = round(float(abs(generator_index + random.randint(xl, xu - 1) - xu) + weight), decimals_precision)
        elif mutation_type == 'generator-and-weight':
            for i, xl, xu in zip(range(size), low, up):
                if random.random() < indpb:
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
                    individual[i] = round(float(abs(generator_index + random.randint(xl, xu - 1) - xu) + weight), decimals_precision)
                    individual[genome_j] = round(generator_index_j + weight_j, decimals_precision)
        return individual,

    def cxTwoPointGAN(self, ind1, ind2, method='average-weights'):
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.
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

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        # print('{}-{}'.format(cxpoint1, cxpoint2))
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

    def create_individual(self, individual_class, size=5, max_generators_index=100):
        weights = self.get_weights_tentative(size)[0]
        generators = [random.randint(0, max_generators_index) for _ in range(size)]
        return individual_class([gen + w for gen, w in zip(generators, weights)])


    def get_all_generators_in_path(self):
        return len([gen for gen in glob.glob('{}*gene*.pkl'.format(self.generators_path))])

    def create_weights_list(self, min_value, max_value, step_size):
        return np.arange(min_value, max_value, step_size)

    def get_genarator_for_ensemble(self, generator_index):
        return '{}mnist-generator-{:03d}.pkl'.format(self.generators_path, generator_index), '{:03d}'.format(generator_index)

    def get_genarators_for_ensemble(self, weight_and_generator_indices):
        return ['{}mnist-generator-{:03d}.pkl'.format(self.generators_path, int(generator_index)) for weight, generator_index in weight_and_generator_indices], \
               ['{:03d}'.format(int(generator_index)) for weight, generator_index in weight_and_generator_indices]

    def get_possible_tentative_weights(self, current_prob, i, size):
        max_prob = 1 * self.precision
        number_of_following_weights = size - i
        if number_of_following_weights==1:
            return [max_prob - current_prob]
        else:
            max_probability = (max_prob - current_prob) / number_of_following_weights
            return self.possible_weights[self.possible_weights <= max_probability]

    def extract_weights(self, weight_and_generator_indices):
        return [round(weight,1) for weight, generator_index in weight_and_generator_indices]


    def get_weights_tentative(self, size):
        result = []
        for i in range(size):
            current_prob = sum(result)
            possible_weights = self.get_possible_tentative_weights(current_prob, i, size)
            result.append(random.choice(possible_weights))
            i += 1
        return np.array(result) / self.precision, len(result)

# weights2 = ga.get_weights_tentative(5)[0]
# print(weights)
# print(sum(weights))
# mut = ga.mutations(weights, 0, 1, 0.2, mutation_type='just-weight-probabilty')
# for i in range(500):
#     weights = ga.get_weights_tentative(5)[0]
#     mut = ga.mutations(weights, 0, 1, 0.2, mutation_type='generator-and-weight')
#     print(mut[0])
#     print(sum(mut[0]))

#
#
# def main():
#
#     def evalOneMax(individual, network_factory, mixture_generator_samples_mode='exact_proportion', fitness_type='TVD'):
#         population = Population(individuals=[], default_fitness=0)
#         weight_and_generator_indices = [math.modf(gen) for gen in individual]
#         generators_paths, sources = ga.get_genarators_for_ensemble(weight_and_generator_indices)
#         tentative_weights = [weight for weight in weight_and_generator_indices]
#         mixture_definition = dict(zip(sources, tentative_weights))
#         for path, source in zip(generators_paths, sources):
#             generator = network_factory.create_generator()
#             generator.net.load_state_dict(torch.load(path))
#             generator.net.eval()
#             population.individuals.append(Individual(genome=generator, fitness=0, source=source))
#
#         dataset = MixedGeneratorDataset(population,
#                                         mixture_definition,
#                                         50000,
#                                         mixture_generator_samples_mode)
#         fid, tvd = score_calc.calculate(dataset)
#
#         if fitness_type=='TVD':
#             return tvd,
#         elif fitness_type=='FID':
#             return fid
#         elif fitness_type == 'FID-TVD':
#             return (fid, tvd),
#
#         #return sum(individual),
#
#     ga = TVDBaserGA()
#     fitness_type='TVD'
#     score_calc = ScoreCalculatorFactory.create()
#     dataloader = 'mnist'
#     network_name = 'four_layer_perceptron'
#     mixture_generator_samples_mode = 'exact_proportion'
#     network_factory = NetworkFactory(network_name, dataloader.n_input_neurons)
#
#     max_generators_index = 10
#     ensemble_size = 5
#
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMin)
#     toolbox = base.Toolbox()
#     toolbox.register("attr_rand", random.uniform, 0, max_generators_index)
#     toolbox.register('individual', ga.create_individual, creator.Individual, size=ensemble_size,
#                      max_generators_index=max_generators_index)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.register("evaluate", evalOneMax, network_factory=network_factory, mixture_generator_samples_mode=mixture_generator_samples_mode, fitness_type=fitness_type)
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mutate", ga.mutate, low=0, up=max_generators_index, indpb=10 / ensemble_size)
#     toolbox.register('crossoverGAN', ga.cxTwoPointGAN)
#     toolbox.register("select", tools.selTournament, tournsize=3)
#
#     pop = toolbox.population(n=100)
#
#     # Evaluate the entire population
#     fitnesses = list(map(toolbox.evaluate, pop))
#     for ind, fit in zip(pop, fitnesses):
#         ind.fitness.values = fit
#
#     # CXPB  is the probability with which two individuals
#     #       are crossed
#     #
#     # MUTPB is the probability for mutating an individual
#     CXPB, MUTPB = 1.0, 1.
#
#     # Extracting all the fitnesses of
#     fits = [ind.fitness.values[0] for ind in pop]
#
#     # Variable keeping track of the number of generations
#     g = 0
#
#     # Begin the evolution
#     while max(fits) < 1000000 and g < 2:
#         # A new generation
#         g = g + 1
#         print("-- Generation %i --" % g)
#
#         # Select the next generation individuals
#         offspring = toolbox.select(pop, len(pop))
#         # Clone the selected individuals
#         offspring = list(map(toolbox.clone, offspring))
#
#         # Apply crossover and mutation on the offspring
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < CXPB:
#                 #toolbox.mate(child1, child2)
#                 toolbox.crossoverGAN(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values
#
#         for mutant in offspring:
#             if random.random() < MUTPB:
#                 print(mutant)
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values
#                 print(mutant)
#
#         # Evaluate the individuals with an invalid fitness
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = map(toolbox.evaluate, invalid_ind)
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit
#
#         pop[:] = offspring
#
#         # Gather all the fitnesses in one list and print the stats
#         fits = [ind.fitness.values[0] for ind in pop]
#
#         length = len(pop)
#         mean = sum(fits) / length
#         sum2 = sum(x * x for x in fits)
#         std = abs(sum2 / length - mean ** 2) ** 0.5
#
#         print("  Min %s" % min(fits))
#         print("  Max %s" % max(fits))
#         print("  Avg %s" % mean)
#         print("  Std %s" % std)
#
#         print([ind for ind in pop])
#
# main()
#
