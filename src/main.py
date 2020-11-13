import argparse
import logging
import os
import re
import sys

import losswise
import torch
import yaml

from helpers.CustomArgumentParser import CustomArgumentParser
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.log_helper import LogHelper
from helpers.population import Population
from helpers.yaml_include_loader import YamlIncludeLoader
from lipizzaner import Lipizzaner
from lipizzaner_client import LipizzanerClient
from lipizzaner_master import LipizzanerMaster, GENERATOR_PREFIX
from training.mixture.score_factory import ScoreCalculatorFactory
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset

import time
from tvd_based_constructor import TVDBasedConstructor
from random_search_ensemble_generator import TVDBasedRandomSearch
from ga_ensemble_generator import TVDBasedGA
from ga_ensemble_generator_free_ensemble_size import TVDBasedSizeFreeGA

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

_logger = logging.getLogger(__name__)


def read_settings(config_filepath):
    with open(config_filepath, 'r') as config_file:
        return yaml.load(config_file, YamlIncludeLoader)


def create_parser():
    def add_config_file(grp, is_required):
        grp.add_argument(
            '--configuration-file',
            '-f',
            type=str,
            required=is_required,
            dest='configuration_file',
            help='YAML configuration file, e.g. configurations/lipizzaner-gan/celeba.yml. Only required on master.')

    parser = CustomArgumentParser(description='Lipizzaner - coevolutionary learning for neural networks')
    parser.add_argument(
        '--cuda-device',
        type=int,
        required=False,
        dest='cuda_device',
        help='If set, the CUDA device with the specific index will be used by PyTorch.')

    subparsers = parser.add_subparsers(dest='task', help='Lipizzaner task to run')

    group_train = subparsers.add_parser('train')
    group_train.add_argument(
        '--distributed',
        action='store_true',
        help='Start as long-running client node. Waits for master '
             'to send experiment configuration, and runs them.')
    group_distributed = group_train.add_mutually_exclusive_group(required='--distributed' in sys.argv)
    group_distributed.add_argument(
        '--master',
        action='store_true',
        help='Start as master node. Runs experiment on all clients, and waits for them to finish.')
    group_distributed.add_argument(
        '--client',
        action='store_true',
        help='Start as long-running client node. Waits for master '
             'to send experiment configuration, and runs them.')
    add_config_file(group_train, '--master' in sys.argv or '--distributed' not in sys.argv)

    group_generate = subparsers.add_parser('generate')
    group_generate.add_argument(
        '--mixture-source',
        type=str,
        dest='mixture_source',
        required=True,
        help='The directory that contains both the generator .pkl files and the yml mixture configuration.')
    group_generate.add_argument(
        '--output',
        '-o',
        type=str,
        dest='output_dir',
        required=True,
        help='The output directory in which the samples will be created in. Will be created if it does not exist yet.')
    group_generate.add_argument(
        '--sample-size',
        type=int,
        dest='sample_size',
        required=True,
        help='The number of samples that will be created.')
    add_config_file(group_generate, True)

    group_score = subparsers.add_parser('score')
    group_score.add_argument(
        '--generator',
        type=str,
        dest='generator_file',
        help='Generator .pkl file.')
    add_config_file(group_score, True)

    group_evaluate = subparsers.add_parser('evaluate')
    group_evaluate.add_argument(
        '--generator',
        type=str,
        dest='generator_file',
        help='Generator .pkl file.')
    group_evaluate.add_argument(
        '--output',
        '-o',
        type=str,
        dest='output_dir',
        required=True,
        help='The output directory in which the samples will be created in. Will be created if it does not exist yet.')
    add_config_file(group_evaluate, True)

    group_optimize_greedy = subparsers.add_parser('optimize-greedy')
    group_optimize_greedy.add_argument(
        '--mode',
        type=str,
        dest='mode',
        help='iterative or random selection')
    group_optimize_greedy.add_argument(
        '--output',
        '-o',
        type=str,
        dest='output_file',
        required=True,
        help='The output file to store the output.')
    group_optimize_greedy.add_argument(
        '--ensemble-max-size',
        '-e',
        type=int,
        dest='ensemble_max_size',
        required=True,
        help='Max size of the ensemble.')
    group_optimize_greedy.add_argument(
        '--n_samples',
        '-ns',
        type=int,
        dest='n_samples',
        required=True,
        help='Max size of the ensemble.')
    add_config_file(group_optimize_greedy, True)

    group_optimize_random = subparsers.add_parser('optimize-random-search')
    group_optimize_random.add_argument(
        '--generations',
        type=int,
        dest='generations',
        help='Number of generations of the random search')
    group_optimize_random.add_argument(
        '--output',
        '-o',
        type=str,
        dest='output_file',
        required=True,
        help='The output file to store the output.')
    group_optimize_random.add_argument(
        '--ensemble-max-size',
        '-e',
        type=int,
        dest='ensemble_max_size',
        required=True,
        help='Max size of the ensemble.')
    group_optimize_random.add_argument(
        '--n_samples',
        '-ns',
        type=int,
        dest='n_samples',
        required=True,
        help='Max size of the ensemble.')
    add_config_file(group_optimize_random, True)

    group_optimize_ga = subparsers.add_parser('optimize-ga')
    group_optimize_ga.add_argument(
        '--generations',
        type=int,
        dest='generations',
        help='Number of generations of the random search')
    group_optimize_ga.add_argument(
        '--population_size',
        type=int,
        dest='population_size',
        help='Number of generations of the random search')
    group_optimize_ga.add_argument(
        '--mutation_probability',
        '-mp',
        type=float,
        dest='mutation_probability',
        help='Number of generations of the random search')
    group_optimize_ga.add_argument(
        '--crossover_probability',
        '-cp',
        type=float,
        dest='crossover_probability',
        help='Number of generations of the random search')
    group_optimize_ga.add_argument(
        '--output',
        '-o',
        type=str,
        dest='output_file',
        required=True,
        help='The output file to store the output.')
    group_optimize_ga.add_argument(
        '--ensemble-max-size',
        '-e',
        type=int,
        dest='ensemble_max_size',
        required=True,
        help='Max size of the ensemble.')
    group_optimize_ga.add_argument(
        '--n_samples',
        '-ns',
        type=int,
        dest='n_samples',
        required=True,
        help='Max size of the ensemble.')
    add_config_file(group_optimize_ga, True)

    return parser


def initialize_settings(args):
    cc = ConfigurationContainer.instance()
    cc.settings = read_settings(args.configuration_file)
    if 'logging' in cc.settings['general'] and cc.settings['general']['logging']['enabled']:
        log_dir = os.path.join(cc.settings['general']['output_dir'], 'log')
        LogHelper.setup(cc.settings['general']['logging']['log_level'], log_dir)
    if cc.is_losswise_enabled:
        losswise.set_api_key(cc.settings['general']['losswise']['api_key'])

    return cc

def optimize_ga(args, cc):
#def optimize_random_search(args, cc):
    def evalOneMax(individual, network_factory, mixture_generator_samples_mode='exact_proportion', n_samples=5000):
        population = Population(individuals=[], default_fitness=0)
        weight_and_generator_indices = [math.modf(gen) for gen in individual]
        generators_paths, sources = ga.get_genarators_for_ensemble(weight_and_generator_indices)
        tentative_weights = ga.extract_weights(weight_and_generator_indices)
        mixture_definition = dict(zip(sources, tentative_weights))
        for path, source in zip(generators_paths, sources):
            generator = network_factory.create_generator()
            generator.net.load_state_dict(torch.load(path))
            generator.net.eval()
            population.individuals.append(Individual(genome=generator, fitness=0, source=source))

        dataset = MixedGeneratorDataset(population,
                                        mixture_definition,
                                        n_samples,
                                        mixture_generator_samples_mode)
        fid, tvd = score_calc.calculate(dataset)
        return fid, tvd,

    def evalOneMax_no_size(individual, network_factory, mixture_generator_samples_mode='exact_proportion', n_samples=5000):
        population = Population(individuals=[], default_fitness=0)
        generators = ga.get_considered_generators(individual)
        weights = ga.get_considered_weights(individual)
        generators_paths, sources = ga.get_genarators_for_ensemble(generators)
        tentative_weights = ga.extract_weights(weights)
        print('Evaluating: ' + str(ga.show_mixture(individual)))
        mixture_definition = dict(zip(sources, tentative_weights))
        for path, source in zip(generators_paths, sources):
            generator = network_factory.create_generator()
            generator.net.load_state_dict(torch.load(path))
            generator.net.eval()
            population.individuals.append(Individual(genome=generator, fitness=0, source=source))

        dataset = MixedGeneratorDataset(population,
                                        mixture_definition,
                                        n_samples,
                                        mixture_generator_samples_mode)
        fid, tvd = score_calc.calculate(dataset)
        return fid, tvd,


    output_file = args.output_file
    ensemble_max_size = args.ensemble_max_size
    generations = args.generations
    population_size = args.population_size
    mutation_probability = args.mutation_probability
    crossover_probability = args.crossover_probability
    n_samples = args.n_samples
    max_generators_index = 200
    ensemble_size = ensemble_max_size

    print('Starting experiments....')
    print('Configuration: ')
    print('Number of generations={}'.format(generations))
    print('Population size={}'.format(population_size))
    print('P_mu={}\nP_cr={}'.format(mutation_probability, crossover_probability))
    print('Max generators={}'.format(max_generators_index))


    #ga = TVDBasedGA()
    min_ensmbe_size = 3
    max_ensmbe_size = ensemble_size
    ga = TVDBasedSizeFreeGA(min_ensmbe_size=min_ensmbe_size, max_ensmbe_size=max_ensmbe_size)

    fitness_type = 'TVD'
    score_calc = ScoreCalculatorFactory.create()
#    dataloader = 'mnist'
#    network_name = 'four_layer_perceptron'
    mixture_generator_samples_mode = 'exact_proportion'
    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)



    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_rand", random.uniform, 0, max_generators_index)
    toolbox.register('individual', ga.create_individual, creator.Individual)
    # toolbox.register('individual', ga.create_individual, creator.Individual, size=ensemble_size,
    #                  max_generators_index=max_generators_index)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax_no_size, network_factory=network_factory,
                     mixture_generator_samples_mode=mixture_generator_samples_mode, n_samples=n_samples)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", ga.mutate, low=0, up=max_generators_index, indpb=1 / ensemble_size)
    toolbox.register('crossoverGAN', ga.cxTwoPointGAN)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    generators_examined = 0

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    tvds = []
    fids = []
    for ind, (fid, tvd) in zip(pop, fitnesses):
        if fitness_type == 'TVD':
            ind.fitness.values = tvd,
        elif fitness_type == 'FID':
            ind.fitness.values = fid,
        elif fitness_type == 'FID-TVD':
            ind.fitness.values = fid, tvd
        tvds.append(tvd)
        fids.append(fid)

    for gen, fid, tvd in zip(pop, fids, tvds):
        print(
            'Generators examined={} - Mixture: {} - FID={}, TVD={}, FIT={}'.format( \
                generators_examined, ga.show_mixture(gen), fid, tvd, gen.fitness.values))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    start_time = time.time()
    # Begin the evolution
    while generators_examined < generations: # g < generations:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_probability:
                # toolbox.mate(child1, child2)
                toolbox.crossoverGAN(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        tvds = []
        fids = []
        for ind, (fid, tvd) in zip(invalid_ind, fitnesses):
            generators_examined += 1
            if fitness_type == 'TVD':
                ind.fitness.values = tvd,
            elif fitness_type == 'FID':
                ind.fitness.values = fid,
            elif fitness_type == 'FID-TVD':
                ind.fitness.values = fid,
            tvds.append(tvd)
            fids.append(fid)
        for gen, fid, tvd in zip(invalid_ind, fids, tvds):
            print(
                'Generators examined={} - Mixture: {} - FID={}, TVD={}'.format( \
                    generators_examined, ga.show_mixture(gen), fid, tvd))

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("FIT  Min %s" % min(fits))
        print("FIT  Max %s" % max(fits))
        print("FIT  Avg %s" % mean)
        print("FIT  Std %s" % std)

        mean = sum(fids) / length
        sum2 = sum(x * x for x in fids)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("FID  Min %s" % min(fids))
        print("FID  Max %s" % max(fids))
        print("FID  Avg %s" % mean)
        print("FID  Std %s" % std)

        mean = sum(tvds) / length
        sum2 = sum(x * x for x in tvds)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("TVD  Min %s" % min(tvds))
        print("TVD  Max %s" % max(tvds))
        print("TVD  Avg %s" % mean)
        print("TVD  Std %s" % std)
        print('Generators examined={}'.format(generators_examined))
        print('Execution time={}'.format(time.time() - start_time))



    print('Finishing execution....')


    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    tvds = []
    fids = []
    for ind, (fid, tvd) in zip(pop, fitnesses):
        print('Mixture: {} - FID = {}, TVD = {}'.format(ind, fid, tvd))
        tvds.append(tvd)
        fids.append(fid)


    length = len(pop)
    mean = sum(fids) / length
    sum2 = sum(x * x for x in fids)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("FID  Min %s" % min(fids))
    print("FID  Max %s" % max(fids))
    print("FID  Avg %s" % mean)
    print("FID  Std %s" % std)

    mean = sum(tvds) / length
    sum2 = sum(x * x for x in tvds)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("TVD  Min %s" % min(tvds))
    print("TVD  Max %s" % max(tvds))
    print("TVD  Avg %s" % mean)
    print("TVD  Std %s" % std)
    print('Generators examined={}'.format(generators_examined))
    print('Execution time={}'.format(time.time() - start_time))

    # print('FID={}'.format(current_fid))
    # print('TVD={}'.format(current_tvd))
    print('Generators examined={}'.format(generators_examined))
    # print('Ensemble: {}'.format(current_mixture_definition))
    print('Execution time={}'.format(time.time() - start_time))




def optimize_random_search(args, cc):
    output_file = args.output_file
    ensemble_max_size = args.ensemble_max_size
    generations=args.generations
    n_samples = args.n_samples
    mixture_generator_samples_mode = 'exact_proportion'

    generators_path = './generators/'
    precision = 100

    constructor = TVDBasedRandomSearch(precision=precision, generators_path=generators_path)
    score_calc = ScoreCalculatorFactory.create()
    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)

    current_tvd = 1.0
    current_fid = 100
    generators_examined = 0
    current_mixture_definition = dict()

    start_time = time.time()
    for generation in range(generations):
        population = Population(individuals=[], default_fitness=0)
        generators_paths, sources = constructor.get_genarators_for_ensemble(ensemble_max_size)
        tentative_weights = constructor.get_weights_tentative(ensemble_max_size)[0]
        mixture_definition = dict(zip(sources, tentative_weights))
        for path, source in zip(generators_paths, sources):
            generator = network_factory.create_generator()
            generator.net.load_state_dict(torch.load(path))
            generator.net.eval()
            population.individuals.append(Individual(genome=generator, fitness=0, source=source))

        dataset = MixedGeneratorDataset(population,
                                        mixture_definition,
                                        n_samples,
                                        mixture_generator_samples_mode)
        fid, tvd = score_calc.calculate(dataset)
        if tvd < current_tvd:
            current_tvd = tvd
            current_fid = fid
            current_mixture_definition = mixture_definition
        generators_examined += 1
        print(
            'Generators examined={} - Mixture: {} - FID={}, TVD={}, FIDbest={}, TVDbest={}'.format( \
                generators_examined, mixture_definition, fid, tvd, current_fid, current_tvd))

    print('Finishing execution....')
    print('FID={}'.format(current_fid))
    print('TVD={}'.format(current_tvd))
    print('Generators examined={}'.format(generators_examined))
    print('Ensemble: {}'.format(current_mixture_definition))
    print('Execution time={}'.format(time.time() - start_time))

    file = open(output_file,'w')
    file.write('FID={}\n'.format(current_fid))
    file.write('TVD={}\n'.format(current_tvd))
    file.write('Generators examined={}'.format(generators_examined))
    file.write('Ensemble: {}\n'.format(current_mixture_definition))
    file.write('Execution time={}\n'.format(time.time()-start_time))
    file.close()


def optimize_greedy(args, cc):
    output_file = args.output_file
    ensemble_max_size = args.ensemble_max_size
    mode = args.mode
    n_samples = args.n_samples
    mixture_generator_samples_mode = 'exact_proportion'
    max_time_without_improvements = 3

    generators_path = './generators/'
    precision = 10
    using_max_size = ensemble_max_size!=0

    constructor = TVDBasedConstructor(precision=precision, generators_path=generators_path, mode=mode)
    score_calc = ScoreCalculatorFactory.create()
    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)
    population = Population(individuals=[], default_fitness=0)
    sources = []

    current_tvd = 1.0
    current_fid = 100
    current_mixture_definition = dict()
    generators_examined = 0

    start_time = time.time()
    while True:
        next_generator_path, source = constructor.get_next_generator_path()#'{}mnist-generator-{:03d}.pkl'.format(generators_path,generator_index)
        generator = network_factory.create_generator()
        generator.net.load_state_dict(torch.load(next_generator_path))
        generator.net.eval()
        #source = '{:03d}'.format(generator_index)
        population.individuals.append(Individual(genome=generator, fitness=0, source=source))
        sources.append(source)
        ensemble_size = len(population.individuals)

        tvd_tentative = 1.0
        mixture_definition_i = dict()

        combinations_of_weights, size = constructor.get_weights_tentative(ensemble_size)
        if size == 0:
            break
        for tentative_mixture_definition in combinations_of_weights:
            mixture_definition = dict(zip(sources, tentative_mixture_definition))
            dataset = MixedGeneratorDataset(population,
                                            mixture_definition,
                                            n_samples,
                                            mixture_generator_samples_mode)
            fid, tvd = score_calc.calculate(dataset)
            if tvd < tvd_tentative:
                tvd_tentative = tvd
                fid_tentative = fid
                mixture_definition_i = mixture_definition
            generators_examined += 1
            print(
                'Generators examined={} - Mixture: {} - FID={}, TVD={}, FIDi={}, TVDi={}, FIDbest={}, TVDbest={}'.format( \
                    generators_examined, mixture_definition, fid, tvd, fid_tentative, tvd_tentative, current_fid, current_tvd))

        if tvd_tentative < current_tvd:
            current_tvd = tvd_tentative
            current_fid = fid_tentative
            current_mixture_definition = mixture_definition_i
            convergence_time = 0
        else:
            sources.pop()
            population.individuals.pop()
            convergence_time += 1

        # print('Ensemble size = {} '.format(len(population.individuals)))
        # _logger.info('Generator loaded from \'{}\' yielded a score of FID={} TVD={}\n{}'.format(next_generator_path,
        #                                                                                     current_fid, current_tvd,
        #                                                                                         current_mixture_definition))
        # print('Generator loaded from \'{}\' yielded a score of FID={} TVD={}\n{}'.format(next_generator_path, current_fid,
        #                                                                             current_tvd, current_mixture_definition))
        #

        if using_max_size and len(sources) == ensemble_max_size:
            break
        else:
            if convergence_time > max_time_without_improvements:
                break

    print('Finishing execution....')
    print('FID={}'.format(current_fid))
    print('TVD={}'.format(current_tvd))
    print('Generators examined={}'.format(generators_examined))
    print('Ensemble: {}'.format(current_mixture_definition))
    print('Execution time={}'.format(time.time() - start_time))

    file = open(output_file, 'w')
    file.write('FID={}\n'.format(current_fid))
    file.write('TVD={}\n'.format(current_tvd))
    file.write('Generators examined={}'.format(generators_examined))
    file.write('Ensemble: {}\n'.format(current_mixture_definition))
    file.write('Execution time={}\n'.format(time.time() - start_time))
    file.close()

def calc_score_optimization(args, cc):
    output_dir = args.output_dir

    score_calc = ScoreCalculatorFactory.create()
    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)

    generator = network_factory.create_generator()
    generator.net.load_state_dict(torch.load(args.generator_file))
    generator.net.eval()
    individual = Individual(genome=generator, fitness=0, source='local')

    dataset = MixedGeneratorDataset(Population(individuals=[individual], default_fitness=0),
                                    {'local': 1.0},
                                    50000,
                                    cc.settings['trainer']['mixture_generator_samples_mode'])

    if output_dir != 'no-images':
        os.makedirs(output_dir, exist_ok=True)
        LipizzanerMaster().save_samples(dataset, output_dir, dataloader)
    inc = score_calc.calculate(dataset)
    _logger.info('Output dir {}'.format(output_dir))
    _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(args.generator_file, inc))
    print('Generator loaded from \'{}\' yielded a score of {}'.format(args.generator_file, inc))


def calc_score(args, cc):
    score_calc = ScoreCalculatorFactory.create()
    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)

    generator = network_factory.create_generator()
    generator.net.load_state_dict(torch.load(args.generator_file))
    generator.net.eval()
    individual = Individual(genome=generator, fitness=0, source='local')

    dataset = MixedGeneratorDataset(Population(individuals=[individual], default_fitness=0),
                                    {'local': 1.0},
                                    50000,
                                    cc.settings['trainer']['mixture_generator_samples_mode'])

    output_dir = os.path.join(cc.output_dir, 'score')
    os.makedirs(output_dir, exist_ok=True)
    LipizzanerMaster().save_samples(dataset, output_dir, dataloader)
    inc = score_calc.calculate(dataset)
    _logger.info('Output dir {}'.format(output_dir))
    _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(args.generator_file, inc))


def generate_samples(args, cc):
    batch_size = 100

    mixture_source = args.mixture_source
    output_dir = args.output_dir
    sample_size = args.sample_size

    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)

    population = Population(individuals=[], default_fitness=0)
    mixture_definition = read_settings(os.path.join(mixture_source, 'mixture.yml'))
    for source, weight in mixture_definition.items():
        path = os.path.join(mixture_source, source)
        generator = network_factory.create_generator()
        generator.net.load_state_dict(torch.load(path))
        generator.net.eval()
        population.individuals.append(Individual(genome=generator, fitness=0, source=source))

    dataset = MixedGeneratorDataset(population,
                                    mixture_definition,
                                    sample_size * batch_size,
                                    cc.settings['trainer']['mixture_generator_samples_mode'])
    os.makedirs(output_dir, exist_ok=True)
    LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)

    dataset = MixedGeneratorDataset(population,
                                    mixture_definition,
                                    50000,
                                    cc.settings['trainer']['mixture_generator_samples_mode'])
    score_calc = ScoreCalculatorFactory.create()
    inc = score_calc.calculate(dataset)
    _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(mixture_source, inc))




if __name__ == '__main__':
    os.environ['TORCH_MODEL_ZOO'] = os.path.join(os.getcwd(), 'output/.models')

    parser = create_parser()
    args = parser.parse_args(args=sys.argv[1:])

    if 'cuda_device' in args and args.cuda_device:
        print('Enforcing usage of CUDA device {}'.format(args.cuda_device))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    if args.task == 'train':
        if args.distributed:
            if args.master:
                initialize_settings(args)
                LipizzanerMaster().run()
            elif args.client:
                LipizzanerClient().run()
        else:
            cc = initialize_settings(args)
            lipizzaner = Lipizzaner()
            lipizzaner.run(cc.settings['trainer']['n_iterations'])

    elif args.task == 'score':
        cc = initialize_settings(args)
        calc_score(args, cc)
    elif args.task == 'generate':
        cc = initialize_settings(args)
        generate_samples(args, cc)
    elif args.task == 'evaluate':
        cc = initialize_settings(args)
        calc_score_optimization(args, cc)
    elif args.task == 'optimize-greedy':
        cc = initialize_settings(args)
        optimize_greedy(args, cc)
    elif args.task == 'optimize-random-search':
        cc = initialize_settings(args)
        optimize_random_search(args, cc)
    elif args.task == 'optimize-ga':
        cc = initialize_settings(args)
        optimize_ga(args, cc)
    else:
        parser.print_help()
