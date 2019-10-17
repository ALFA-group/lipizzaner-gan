import argparse
import logging
import os
import re
import sys

from collections import OrderedDict
import numpy as np
import time
from helpers.pytorch_helpers import noise
from training.mixture.mixed_generator_dataset import MixedGeneratorDatasetES

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

    group_generate = subparsers.add_parser('optimize')
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
    score_calc = ScoreCalculatorFactory.create()
    inc = score_calc.calculate(dataset)
    _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, inc))
    os.makedirs(output_dir, exist_ok=True)
    LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)

def individual_scores(population, mixture_definition, sample_size, batch_size, z_noise, cc):
    weight_vector_length = len(mixture_definition)
    # Uniform weights distribution
    transformed = (1.0/float(weight_vector_length)) * np.ones(weight_vector_length)
    new_mixture_weights_generators = OrderedDict(zip(mixture_definition.keys(), transformed))
    print('Individual score')

    
    _logger.info('Mixture {}'.format(transformed))

    dataset = MixedGeneratorDatasetES(population,
                                    new_mixture_weights_generators,
                                    sample_size * batch_size,
                                    cc.settings['trainer']['mixture_generator_samples_mode'], z_noise)

    #LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)

    score_calc = ScoreCalculatorFactory.create()
    inc = score_calc.calculate(dataset)
    _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, inc))
    print('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, inc))
    print('Runing information: {},{}'.format(-1000,inc))

    # Each generator by itself
    for i in range(weight_vector_length):
        transformed = np.zeros(weight_vector_length)
        transformed[i] = 1.0
        new_mixture_weights_generators = OrderedDict(zip(mixture_definition.keys(), transformed))
        _logger.info('Mixture {}'.format(transformed))

        dataset = MixedGeneratorDatasetES(population,
                                        new_mixture_weights_generators,
                                        sample_size * batch_size,
                                        cc.settings['trainer']['mixture_generator_samples_mode'], z_noise)

        #LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)

        score_calc = ScoreCalculatorFactory.create()
        inc = score_calc.calculate(dataset)[0]
        _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, inc))
        print('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, inc))
        print('Runing information: {},{}'.format(-1000,inc))



def optimize_ensamble(args, cc):
    batch_size = 100

    #generations = 20
    execution_time = 30 * 60 # in seconds
    mixture_source = args.mixture_source
    output_dir = args.output_dir
    sample_size = args.sample_size
    mixture_sigma = cc.settings['trainer']['params']['mixture_sigma']

    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)

    population = Population(individuals=[], default_fitness=0)
    mixture_definition = read_settings(os.path.join(mixture_source, 'mixture.yml'))
    if len(mixture_definition) < 2:
        print('No weights to optimize')
        _logger.info('No weights to optimize')
        sys.exit(1)
   
    _logger.info('Mixture {}'.format(mixture_definition))
    print('Mixture {}'.format(mixture_definition))
    print(len(mixture_definition.items()))
    for source, weight in mixture_definition.items():
        path = os.path.join(mixture_source, source)
        generator = network_factory.create_generator()
        generator.net.load_state_dict(torch.load(path))
        generator.net.eval()
        population.individuals.append(Individual(genome=generator, fitness=0, source=source))

    z_noise = noise(sample_size * batch_size, population.individuals[0].genome.data_size)

    dataset = MixedGeneratorDatasetES(population,
                                    mixture_definition,
                                    sample_size * batch_size,
                                    cc.settings['trainer']['mixture_generator_samples_mode'], z_noise)

    #LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)

    score_calc = ScoreCalculatorFactory.create()
    score = score_calc.calculate(dataset)[0]
    _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, score))
    print('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, score))
    print('Runing information:{},{}'.format(-1, score))

    weight_vector_length = len(mixture_definition)
    # Uniform weights distribution
    transformed = (1.0/float(weight_vector_length)) * np.ones(weight_vector_length)
    mixture_weights_generators = OrderedDict(zip(mixture_definition.keys(), transformed))
    print('Mutated Mixture {}'.format(transformed))

    dataset = MixedGeneratorDatasetES(population, mixture_definition, sample_size * batch_size, cc.settings['trainer']['mixture_generator_samples_mode'], z_noise)
    score_calc = ScoreCalculatorFactory.create()
    score = score_calc.calculate(dataset)[0]
    _logger.info('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, score))
    print('Generator loaded from \'{}\' yielded a score of {}'.format(args.mixture_source, score))
    print('Runing information:{},{}'.format(-1, score))
    mixture_definition = mixture_weights_generators

    g = 0
    start_time = time.time()
    while(execution_time > (time.time() - start_time)):
    #for g in range(generations):
        # Mutate mixture weights
        z = np.random.normal(loc=0, scale=mixture_sigma, size=len(mixture_definition))
        transformed = np.asarray([value for _, value in mixture_definition.items()])
        transformed += z
        # Don't allow negative values, normalize to sum of 1.0
        transformed = np.clip(transformed, 0, None)
        transformed /= np.sum(transformed)
        _logger.info('Mutated Mixture {}'.format(transformed))
        print('Mutated Mixture {}'.format(transformed))

        new_mixture_weights_generators = OrderedDict(zip(mixture_definition.keys(), transformed))

        #dataset_before_mutation = MixedGeneratorDataset(population,
        #                                                mixture_definition,
        #                                                sample_size * batch_size,
        #                                                cc.settings['trainer']['mixture_generator_samples_mode'])
        #dataset_after_mutation = MixedGeneratorDataset(population,
        dataset = MixedGeneratorDatasetES(population, new_mixture_weights_generators,
                                                        sample_size * batch_size,
                                                        cc.settings['trainer']['mixture_generator_samples_mode'], z_noise)

        if score_calc is not None:
            # logger.info('Calculating FID/inception score.')
            #
            # score_before_mutation = score_calc.calculate(dataset_before_mutation)[0]
            # _logger.info('Score before mutation: {}.'.format(score_before_mutation))
            score_after_mutation = score_calc.calculate(dataset)[0]
            _logger.info('Score after mutation: {}.'.format(score_after_mutation))
            print('Generation: {} \tScore after mutation: {}.'.format(g, score_after_mutation))


            # For fid the lower the better, for inception_score, the higher the better
            if (score_after_mutation < score and score_calc.is_reversed) \
                    or (score_after_mutation > score and (not score_calc.is_reversed)):
                # Adopt the mutated mixture_weights only if the performance after mutation is better
                mixture_definition = new_mixture_weights_generators
                score = score_after_mutation
            # else:
            #     # Do not adopt the mutated mixture_weights here
            #     score = score_before_mutation

        _logger.info('Generation: {} \tScore: {}'.format(g, score))
        print('Runing information:{},{}'.format(g, score))
        g += 1

        #LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)

    _logger.info('Generations: {} \tFinal score: {} \tRun time: {} minutes'.format(g, score, (time.time() - start_time)/60))
    print('Generations: {} \tFinal score: {} \tRun time: {} minutes'.format(g, score, (time.time() - start_time)/60))

    individual_scores(population, mixture_definition, sample_size, batch_size, z_noise, cc)
    #os.makedirs(output_dir, exist_ok=True)
    #LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)

if __name__ == '__main__':
    os.environ['TORCH_MODEL_ZOO'] = os.path.join(os.getcwd(), 'output/.models')

    parser = create_parser()
    args = parser.parse_args(args=sys.argv[1:])

    if 'cuda_device' in args and args.cuda_device:
        print('Enforcing usage of CUDA device {}'.format(args.cuda_device))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    print(args.task)

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

    elif args.task == 'optimize':
        cc = initialize_settings(args)
        optimize_ensamble(args, cc)

    else:
        parser.print_help()
