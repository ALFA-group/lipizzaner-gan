"""
Author: Jamal Toutouh (toutouh@mit.edu) - www.jamal.es

This code is part of the research of our paper "Re-purposing Heterogeneous Generative Ensembles with Evolutionary
Computation" presented during GECCO 2020 (https://doi.org/10.1145/3377930.3390229)

greedy_for_ensemble_generator.py contains the code of the greedy algorithms defined to create ensembles of (GAN)
 generators.
"""
import numpy as np
import random
from itertools import product
import glob
import torch
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.population import Population
from training.mixture.score_factory import ScoreCalculatorFactory
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset

import sys
import time


class GreedyEnsembleGenerator:

    fitness_types = ['tvd', 'fid', 'tvd-fid'] # Currently supported GAN metrics. 'tvd-fid' for future multi-objective

    def __init__(self, dataset, ensemble_max_size=5, max_time_without_improvements=3, precision=10, generators_prefix='mnist-generator',
                 generators_path='./generators/', mode='iterative', output_file=''):
        self.list_of_weights = []
        self.list_of_generators = []
        self.precision = precision
        self.possible_weights = self.create_weights_list(0, self.precision+1, 1)
        self.ensemble_max_size = ensemble_max_size
        self.generator_index = 0
        self.generators_path = generators_path
        self.generators_prefix = generators_prefix
        self.generators_in_path = self.get_maximum_generators_index(generators_path, generators_prefix)
        self.mode = mode
        self.max_time_without_improvements = max_time_without_improvements
        self.configure_lipizzaner(dataset)
        self.create_generators_list()
        self.output_file = output_file

    def configure_lipizzaner(self, dataset):
        """It configures the parameters required to use Lipizzaner to evaluate the ensembles.
        :parameter dataset: dataset addressed in the problem
        """
        cuda_availability = torch.cuda.is_available()
        cc = ConfigurationContainer.instance()
        settings = {'trainer': {'params': {'score': {'type': 'fid'}}},
                    'dataloader': {'dataset_name': dataset},
                    'master': {'cuda': cuda_availability},
                    'network': {'loss': 'bceloss',
                                'name': 'four_layer_perceptron'},
                    'general': {'distribution': {'auto_discover': 'False'},
                                'output_dir': './output',
                                'num_workers': 0}
                    }
        cc.settings = settings
        data_loader = cc.create_instance(dataset)
        self.mixture_generator_samples_mode = 'exact_proportion'
        self.score_calc = ScoreCalculatorFactory.create()
        self.network_factory = cc.create_instance('four_layer_perceptron', data_loader.n_input_neurons)

    def create_generators_list(self):
        self.generators_list = list(range(self.generators_in_path))
        if self.mode == 'random':
            random.shuffle(self.generators_list)

    def get_all_generators_in_path(self):
        return len([gen for gen in glob.glob('{}*gene*.pkl'.format(self.generators_path))])

    def create_weights_list(self, min_value, max_value, step_size):
        return np.arange(min_value, max_value, step_size)

    def get_next_generator_path(self):
        path, source = '', ''
        if self.generator_index < self.generators_in_path:
            path = '{}/{}-{:03d}.pkl'.format(self.generators_path, self.generators_prefix,
                                             int(self.generators_list[self.generator_index]))
            source = '{:03d}'.format(int(self.generators_list[self.generator_index]))
            self.generator_index += 1
        return path, source

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

    def get_weights_tentative(self, size):
        tentative_weights = list(product(list(self.possible_weights), repeat=size))
        result = []
        for w in tentative_weights:
            if sum(w)==self.precision:
                result.append(list(w))
        return np.array(result)/self.precision, len(result)

    def get_next_generator(self, criterion):
        return None

    def show_file_screen(self, text, output_file=''):
        if output_file == '':
            print(text)
        else:
            file = open(output_file, 'a')
            file.write(text)
            file.close()

    def show_experiment_configuration(self):
        text = 'Greedy configuration: '
        text += 'mode={}, '.format(self.mode)
        text += 'ensemble size={}, '.format(self.ensemble_max_size)
        text += 'max iterations without improvement={}, '.format(self.max_time_without_improvements)
        text += 'precision={}, '.format(self.precision)
        text += 'output_file={} \n'.format(self.output_file)
        self.show_file_screen(text)
        if self.output_file != '': self.show_file_screen(text, self.output_file)

    def create_ensemble(self):
        n_samples = 50000
        using_max_size = self.ensemble_max_size != 0

        population = Population(individuals=[], default_fitness=0)
        sources = []

        current_tvd = 1.0
        current_fid = 100
        current_mixture_definition = dict()
        generators_examined = 0

        self.show_experiment_configuration()

        start_time = time.time()
        while True:
            next_generator_path, source = self.get_next_generator_path()
            if next_generator_path == '':
                text = 'Warning: \n'
                text += 'No more generators to be examined to be added to the ensemble. \n'
                text += 'Generators examined: {}\n'.format(generators_examined)
                self.show_file_screen(text)
                if self.output_file != '': self.show_file_screen(text, self.output_file)
                break
            generator = self.network_factory.create_generator()
            generator.net.load_state_dict(torch.load(next_generator_path, map_location='cpu'))
            generator.net.eval()

            population.individuals.append(Individual(genome=generator, fitness=0, source=source))
            sources.append(source)
            ensemble_size = len(population.individuals)

            tvd_tentative = 1.0
            mixture_definition_i = dict()

            combinations_of_weights, size = self.get_weights_tentative(ensemble_size)
            if size == 0:
                break

            for tentative_mixture_definition in combinations_of_weights:
                mixture_definition = dict(zip(sources, tentative_mixture_definition))
                dataset = MixedGeneratorDataset(population,
                                                mixture_definition,
                                                n_samples,
                                                self.mixture_generator_samples_mode)
                fid, tvd = self.score_calc.calculate(dataset)
                if tvd < tvd_tentative:
                    tvd_tentative = tvd
                    fid_tentative = fid
                    mixture_definition_i = mixture_definition
                generators_examined += 1
                text = 'Generators examined={} - Mixture: {} - FID={}, TVD={}, FIDi={}, TVDi={}, FIDbest={}, ' \
                       'TVDbest={}'.format(generators_examined, mixture_definition, fid, tvd, fid_tentative,
                                           tvd_tentative, current_fid, current_tvd)
                self.show_file_screen(text)
                if self.output_file != '': self.show_file_screen(text+ '\n', self.output_file)

            if tvd_tentative < current_tvd:
                current_tvd = tvd_tentative
                current_fid = fid_tentative
                current_mixture_definition = mixture_definition_i
                convergence_time = 0
            else:
                sources.pop()
                population.individuals.pop()
                convergence_time += 1

            if using_max_size and len(sources) == self.ensemble_max_size:
                break
            else:
                if self.max_time_without_improvements!= 0 and convergence_time > self.max_time_without_improvements:
                    break

        text = 'Finishing execution....\n'
        text += 'FID={}'.format(current_fid)
        text += 'TVD={}'.format(current_tvd)
        text += 'Generators examined={}'.format(generators_examined)
        text += 'Ensemble: {}'.format(current_mixture_definition)
        text += 'Execution time={} \n'.format(time.time() - start_time)

        self.show_file_screen(text)
        if self.output_file != '': self.show_file_screen(text, self.output_file)

# dataset = 'mnist'
# precision=10
# mode='random'
# ensemble_max_size = 3
# greedy = GreedyEnsembleGenerator(dataset, ensemble_max_size, precision, generators_prefix='mnist-generator', generators_path='./mnist-generators/',
#                  mode=mode)
#
# greedy.create_ensemble()