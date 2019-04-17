from data import network_data_loader as ndl
from networks import network_factory as nf
from helpers.configuration_container import ConfigurationContainer

import torch
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

_logger = logging.getLogger(__name__)

def read_settings(config_filepath):
    with open(config_filepath, 'r') as config_file:
        return yaml.load(config_file, YamlIncludeLoader)
def create_parser():
    print("creating parser")
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

    return parser
def initialize_settings(args):
    print("initializing settings")
    cc = ConfigurationContainer.instance()
    cc.settings = read_settings(args.configuration_file)

    if 'logging' in cc.settings['general'] and cc.settings['general']['logging']['enabled']:
        log_dir = os.path.join(cc.settings['general']['output_dir'], 'log')
        LogHelper.setup(cc.settings['general']['logging']['log_level'], log_dir)
    if cc.is_losswise_enabled:
        losswise.set_api_key(cc.settings['general']['losswise']['api_key'])
    print("done initializing settings")
    return cc
def calc_score(args, cc):
    print("calculating scores")

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
    print("generating samples")
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


os.environ['TORCH_MODEL_ZOO'] = os.path.join(os.getcwd(), 'output/.models')

parser = create_parser()
args = parser.parse_args(args=sys.argv[1:])
initialize_settings(args)


sequences = ndl.generate_random_sequences(100)
def fake_loss(*args):
    return 0

rnn_factory = nf.RNNFactory(4, loss_function=fake_loss)
# perceptron = nf.FourLayerPerceptronFactory(4, loss_function=fake_loss)

generator = rnn_factory.create_generator()
# print(generator.net)
discriminator = rnn_factory.create_discriminator()
# print(discriminator.net)

dataloader = ndl.NetworkDataLoader()

good_sequences = torch.from_numpy(sequences)

data = dataloader.load()

for batch in data:
    # print(generator.net(batch))
    # x = generator.compute_loss_against(discriminator, batch)
    # print(x[1].shape)

    y = discriminator.compute_loss_against(generator, batch)
    sys.exit()

    print(len(x))
    print((x[0]))
    print(x[1].shape)


    sys.exit()
