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

class TVDBasedSizeFreeGA:
    def __init__(self, precision=100, generators_path='./generators/', min_ensmbe_size=3, max_ensmbe_size=8):
        self.generators_path = generators_path
        self.generators_in_path = self.get_all_generators_in_path()
        #self.generators_in_path = 6
        self.precision = precision
        self.possible_weights = self.create_weights_list(0, self.precision+1, 1)
        self.min_ensmbe_size = min_ensmbe_size
        self.max_ensmbe_size = self.generators_in_path  if max_ensmbe_size > self.generators_in_path else max_ensmbe_size

    def initialize_generators_index(self, max_generator_elements):
        generators_inex =list(range(max_generator_elements))
        random.shuffle(generators_inex)
        return generators_inex

    def initialize_weights(self, max_generator_elements):
        return [random.randint(0,10) for i in range(max_generator_elements)]

    def create_individual(self, individual_class):
        ensemble_size = [random.randint(self.min_ensmbe_size, self.max_ensmbe_size)]
        generators_index = self.initialize_generators_index(self.generators_in_path)
        weights = self.initialize_weights(self.generators_in_path)
        individual = self.normalize_weights(ensemble_size+generators_index+weights)
        return individual_class(individual)

    def get_all_generators_in_path(self):
        return len([gen for gen in glob.glob('{}*gene*.pkl'.format(self.generators_path))])

    def create_weights_list(self, min_value, max_value, step_size):
        return np.arange(min_value, max_value, step_size)


    def mutate(self, individual, low, up, indpb):
        prob = random.random()
        if prob < 0.25:
            mutation_type = 'just-weight-probabilty'
        elif prob < 0.50:
            mutation_type = 'just-generator-index'
        elif prob < 0.75:
            mutation_type = 'generator-and-weight'
        else:
            mutation_type = 'just-ensemble-size'
        individual, = self.mutations(individual, low, up, indpb, mutation_type=mutation_type)
        return individual,

    def mutations(self, individual, low, up, indpb, mutation_type='generator-and-weight'):
        #print('Lets mutate - Probabolity:' + str(indpb)+ ' Mutation type: ' + mutation_type)
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

        if mutation_type == 'just-ensemble-size':
            if random.random() < indpb:
                ensemble_size_range = self.max_ensmbe_size - self.min_ensmbe_size
                self.normalize_weights(individual)

        elif mutation_type == 'just-weight-probabilty':
            weights = self.get_considered_weights(individual)
            for i in range(len(weights)):
                if random.random() < indpb:
                    weights[i] = random.randint(0, self.precision)
            self.set_indiviudal_weights(individual, weights)
            individual = self.normalize_weights(individual)

        elif mutation_type == 'just-generator-index':
            generators = self.get_considered_generators(individual)
            if len(generators) >= self.generators_in_path:
                new_idx = random.randint(1, self.generators_in_path)
            else:
                new_idx = random.randint(len(generators) + 1, self.generators_in_path)
            for i in range(len(generators)):
                if random.random() < indpb:
                    individual = self.swap_genes(individual, i+1, new_idx)

        elif mutation_type == 'generator-and-weight':
            weights = self.get_considered_weights(individual)
            if len(weights) >= self.generators_in_path:
                new_idx = random.randint(1, self.generators_in_path)
            else:
                new_idx = random.randint(len(weights) + 1, self.generators_in_path)
            for i in range(len(weights)):
                if random.random() < indpb:
                    weights[i] = random.randint(0, self.precision)
                    individual = self.swap_genes(individual, i + 1, new_idx)
            self.set_indiviudal_weights(individual, weights)
            individual = self.normalize_weights(individual)

        return individual,


    def get_individual_generators(self, ind):
        return ind[1:self.generators_in_path + 1]

    def get_considered_generators(self, ind):
        return ind[1: ind[0]+1]

    def get_not_considered_generators(self, ind):
        return ind[ind[0]+1: self.generators_in_path + 1]

    def get_individual_weigths(self, ind):
        return ind[self.generators_in_path+1 : 2*self.generators_in_path+1]

    def get_considered_weights(self, ind):
        return ind[self.generators_in_path+1: self.generators_in_path+1+ind[0]]

    def get_not_considered_weights(self, ind):
        return ind[self.generators_in_path+ind[0]+1: 2*self.generators_in_path+1]

    def set_indiviudal_weights(self, ind, weights):
        ind[self.generators_in_path+1 : self.generators_in_path+1+len(weights)] = weights
        return

    def normalize_weights(self, ind):
        total_sum = sum(ind[self.generators_in_path+1: self.generators_in_path+1 + ind[0]])
        if total_sum == 0: # Work around
            for i in range(self.generators_in_path + 1, self.generators_in_path + 1 + ind[0]):
                ind[i] = random.randint(10,20)
        else:
            for i in range(self.generators_in_path+1, self.generators_in_path + 1 + ind[0]):
                ind[i] = int(self.precision * ind[i] / total_sum)
        ind[self.generators_in_path + ind[0]] = self.precision - sum(ind[self.generators_in_path+1: self.generators_in_path+ind[0]])
        return ind

#------------------------------------------------------------------
# For Crossover

    def swap_genes(self, ind, idx1, idx2):
        aux = ind[idx1], ind[idx2]
        ind[idx2], ind[idx1] = aux
        return ind

    def cross_generators_genomes(self,ind1, ind2, cxpoint1, cxpoint2, offset):
        # new_genomes = ind2[cxpoint1:cxpoint2]
        # for dest, gen in zip(range(cxpoint1, cxpoint2), new_genomes):
        #     idx = ind1.index(gen)
        #     ind1[idx] = ind1[dest]
        #     ind1[dest] = gen
        generators_ind1 = self.get_individual_generators(ind1)
        generators_ind2 = self.get_individual_generators(ind2)
        for gen in range(cxpoint1+offset, cxpoint2+offset):
            idx1 = generators_ind1.index(ind2[gen])
            idx2 = generators_ind2.index(ind1[gen])
            ind1 = self.swap_genes(ind1, gen, idx1+offset)
            ind2 = self.swap_genes(ind2, gen, idx2+offset)
        return ind1, ind2


    def show_mixture(self, ind):
        line = 'Size = ' + str(ind[0]) + ' - Generator = ['
        for i in range(1, ind[0]+1):
            line += str(ind[i]) + '-' + str(round(ind[self.generators_in_path+i]/ self.precision, 2)) + ', '
        return line[:-2] + ']'




    def cross_weights_genomes(self, ind1, ind2, cxpoint1, cxpoint2, offset):
        cxpoint1 += offset
        cxpoint2 += offset
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
                 = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        return ind1, ind2



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
        size = max(ind1[0], ind2[0])
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Cross generators index --> Offset = 1
        ind1, ind2 = self.cross_generators_genomes(ind1, ind2, cxpoint1, cxpoint2, 1)

        # Cross weights index --> Offset = self.generators_in_path+1
        ind1, ind2 = self.cross_weights_genomes(ind1, ind2, cxpoint1, cxpoint2, self.generators_in_path+1)
        ind1 = self.normalize_weights(ind1)
        ind2 = self.normalize_weights(ind2)

        return ind1, ind2

    def get_genarator_for_ensemble(self, generator_index):
        return '{}mnist-generator-{:03d}.pkl'.format(self.generators_path, generator_index), '{:03d}'.format(generator_index)

    def get_genarators_for_ensemble(self, generator_indices):
        return ['{}mnist-generator-{:03d}.pkl'.format(self.generators_path, int(generator_index)) for generator_index in generator_indices], \
               ['{:03d}'.format(int(generator_index)) for generator_index in generator_indices]

    def get_possible_tentative_weights(self, current_prob, i, size):
        max_prob = 1 * self.precision
        number_of_following_weights = size - i
        if number_of_following_weights==1:
            return [max_prob - current_prob]
        else:
            max_probability = (max_prob - current_prob) / number_of_following_weights
            return self.possible_weights[self.possible_weights <= max_probability]

    def extract_weights(self, weights):
        return [round(weight/self.precision,2) for weight in weights]

    def get_weights_tentative(self, size):
        result = []
        for i in range(size):
            current_prob = sum(result)
            possible_weights = self.get_possible_tentative_weights(current_prob, i, size)
            result.append(random.choice(possible_weights))
            i += 1
        return np.array(result) / self.precision, len(result)

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
#
#     min_ensmbe_size = 3
#     max_ensmbe_size = 8
#
#     ga = TVDBasedSizeFreeGA(min_ensmbe_size=min_ensmbe_size, max_ensmbe_size=max_ensmbe_size)
#     # score_calc = ScoreCalculatorFactory.create()
#     # dataloader = 'mnist'
#     # network_name = 'four_layer_perceptron'
#     # mixture_generator_samples_mode = 'exact_proportion'
#     # network_factory = NetworkFactory(network_name, dataloader.n_input_neurons)
#
#     max_generators_index = 10
#     ensemble_size = 5
#
#
#
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMin)
#     toolbox = base.Toolbox()
#     toolbox.register("attr_rand", random.uniform, 0, max_generators_index)
#     toolbox.register('individual', ga.create_individual, creator.Individual)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.register("evaluate", evalOneMax, network_factory=network_factory, mixture_generator_samples_mode=mixture_generator_samples_mode, fitness_type=fitness_type)
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mutate", ga.mutate, low=0, up=max_generators_index, indpb=10 / ensemble_size)
#     toolbox.register('crossoverGAN', ga.cxTwoPointGAN)
#     toolbox.register("select", tools.selTournament, tournsize=3)
#
#     pop = toolbox.population(n=2)
#
#     for ind in pop:
#         print('Individual: ' + str(ind))
#         print('Generators: '  + str(ga.get_individual_generators(ind)))
#         print('Weights: ' + str(ga.get_individual_weigths(ind)))
#
#     print(ga.cxTwoPointGAN(pop[0], pop[1]))
#     print(ga.mutations(pop[0], 0,1,1.0,'generator-and-weight'))
#
#     sys.exit(0)
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
