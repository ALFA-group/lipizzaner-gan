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

#from enesmble_optimization.ga_for_ensemble_generator import GAEnsembleGenerator
#from enesmble_optimization.greedy_for_ensemble_generator import GreedyEnsembleGenerator

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

    group_ensemble = subparsers.add_parser('ensemble-generator')
    group_ensemble.add_argument(
        '--generators',
        type=str,
        dest='generators_folder',
        help='The directory that contains both the generator .pkl files.')
    group_ensemble.add_argument(
        '--generators_prefix',
        type=str,
        dest='generators_prefix',
        required=True,
        help='The prefix given to the generator .pkl files to be used to generate the ensembles. If not is given no '
             'pattern is applied.')
    group_ensemble.add_argument(
        '--output',
        '-o',
        type=str,
        dest='output_file',
        required=False,
        help='The output file with the results. Will be created if it does not exist yet.')
    add_config_file(group_ensemble, True)


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
    cc.settings['general']['distribution']['client_id'] = 0
    dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = network_factory.create_generator()
    generator.net.load_state_dict(torch.load(args.generator_file, map_location=device))
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

    dataset_name = cc.settings['dataloader']['dataset_name']
    if dataset_name == 'mnist_labels':
        dataloader = cc.create_instance(dataset_name, cc.settings['dataloader']['labels'], cc.settings['dataloader']['labels_per_cell'])
    else:
        dataloader = cc.create_instance(dataset_name)
    network_factory = cc.create_instance(cc.settings['network']['name'], dataloader.n_input_neurons)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    population = Population(individuals=[], default_fitness=0)
    mixture_definition = read_settings(os.path.join(mixture_source, 'mixture.yml'))
    for source, weight in mixture_definition.items():
        path = os.path.join(mixture_source, source)
        generator = network_factory.create_generator()
        generator.net.load_state_dict(torch.load(path, map_location=device))
        generator.net.eval()
        population.individuals.append(Individual(genome=generator, fitness=0, source=source))

    dataset = MixedGeneratorDataset(population,
                                    mixture_definition,
                                    sample_size * batch_size,
                                    cc.settings['trainer']['mixture_generator_samples_mode'])
    os.makedirs(output_dir, exist_ok=True)
    LipizzanerMaster().save_samples(dataset, output_dir, dataloader, sample_size, batch_size)


def ensemble_optimization(args, cc):
    generators_path = args.generators_folder
    generators_prefix = args.generators_prefix
    output_file = args.output_file if not args.output_file is None else ''

    algorithm =  cc.settings['ensemble_optimization']['algorithm']
    mode = cc.settings['ensemble_optimization']['params']['mode']
    dataset = cc.settings['dataloader']['dataset_name']

    if algorithm == 'ga':
        if mode == 'nreo-gen':
            min_ensemble_size = cc.settings['ensemble_optimization']['params']['min_ensemble_size']
            max_ensemble_size = cc.settings['ensemble_optimization']['params']['max_ensemble_size']
        elif mode == 'reo-gen':
            max_ensemble_size = min_ensemble_size = cc.settings['ensemble_optimization']['params']['ensemble_size']
        else:
            print('Error: Evolutionary algorithm generators ensemble creator requires selecting a mode: reo-gen or '
                  'nreo-gen ')
            sys.exit(-1)
        fitness_metric = cc.settings['ensemble_optimization']['params']['fitness_metric']
        crossover_probability = cc.settings['ensemble_optimization']['params']['crossover_probability']
        mutation_probability = cc.settings['ensemble_optimization']['params']['mutation_probability']
        number_of_fitness_evaluations = cc.settings['ensemble_optimization']['params']['n_fitness_evaluations']
        population_size = cc.settings['ensemble_optimization']['params']['population_size']
        number_of_generations = cc.settings['ensemble_optimization']['params']['n_generations']
        show_info_iteration = cc.settings['ensemble_optimization']['frequency_show_information']
        ga = GAEnsembleGenerator(dataset, min_ensemble_size, max_ensemble_size, generators_path, generators_prefix,
                                 fitness_metric,mode, population_size, number_of_generations,
                                 number_of_fitness_evaluations,
                                 mutation_probability, crossover_probability, show_info_iteration, output_file)

        ga.evolutionary_loop()

    elif algorithm == 'greedy':
        if mode in ['random', 'iterative']:
            ensemble_max_size = cc.settings['ensemble_optimization']['params']['max_ensemble_size']
            max_time_without_improvements = cc.settings['ensemble_optimization']['params']['max_iterations_without_improvemets']
            precision = 10
            greedy = GreedyEnsembleGenerator(dataset, ensemble_max_size, max_time_without_improvements, precision,
                                             generators_prefix, generators_path, mode, output_file)
            greedy.create_ensemble()
        else:
            print('Error: Greedy generators ensemble creator requires selecting a mode: iterative or random')
            sys.exit(-1)
    else:
        print('Error: Select an algorithm between ga and greedy.')
        sys.exit(-1)


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'output/.models')

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

    elif args.task == 'ensemble-generator':
        cc = initialize_settings(args)
        ensemble_optimization(args, cc)
    else:
        parser.print_help()
