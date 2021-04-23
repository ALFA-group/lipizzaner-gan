import logging
import os
from abc import abstractmethod, ABC

from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import noise
from distribution.state_encoder import StateEncoder
from distribution.neighbourhood import Neighbourhood
from helpers.yaml_include_loader import YamlIncludeLoader
from helpers.population import Population, TYPE_GENERATOR, TYPE_DISCRIMINATOR
from helpers.individual import Individual

import yaml
import gzip
from datetime import datetime
import torch
GENERATOR_PREFIX = 'generator-'
DISCRIMINATOR_PREFIX = 'discriminator-'


class NeuralNetworkTrainer(ABC):

    _logger = logging.getLogger(__name__)

    """
    Abstract base class for neural network training modules, cannot be instanced.
    """
    def __init__(self, dataloader, network_factory):
        self.dataloader = dataloader
        self.network_factory = network_factory
        self.cc = ConfigurationContainer.instance()
        self.neighbourhood = Neighbourhood.instance()

        checkpoint_dir = self.get_checkpoint_dir()
        if os.path.isdir(checkpoint_dir):
            self._logger.info("Reading checkpoint")
            this_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-{}.yml'.format(self.neighbourhood.cell_number))
            try:
                with gzip.open(this_checkpoint_path, 'rt', encoding='UTF-8') as checkpoint_file:
                    checkpoint = yaml.load(checkpoint_file, YamlIncludeLoader)
                    self._logger.info("Finished reading checkpoint")
                    self._logger.info("Generator iter: {}".format(checkpoint['iteration']))
                    self.population_gen, self.population_dis = self.parse_populations(checkpoint)
            except:
                self.start_iter = 0
                self.population_gen, self.population_dis = self.initialize_populations()
                self._logger.info("Failed to read checkpoint, starting over")
        else:
            self.start_iter = 0
            self.population_gen, self.population_dis = self.initialize_populations()

    """
    helper function to parse the checkpoint file and create generator and disc populations
    """
    def parse_populations(self, checkpoint):
        pops = ()
        l = [('generators', TYPE_GENERATOR, self.network_factory.create_generator),
             ('discriminators', TYPE_DISCRIMINATOR, self.network_factory.create_discriminator)]
        iteration = checkpoint['iteration']
        for pop,pop_type,create_genome in l:
            self._logger.info("Began parsing {}".format(pop))
            default_fitness = checkpoint[pop]['default_fitness']
            learning_rate = checkpoint[pop]['learning_rate']
            self.start_iter = iteration
            individuals = []
            for indiv in checkpoint[pop]['individuals']:
                new_indiv = Individual.decode(create_genome,
                                              indiv['parameters'],
                                              is_local=True,
                                              learning_rate=learning_rate,
                                              optimizer_state=StateEncoder.decode(indiv['state_encoder']),
                                              source=indiv['source'],
                                              id=indiv['id'],
                                              iteration=iteration)
                individuals.append(new_indiv)
            new_pop = Population(individuals, default_fitness, pop_type)
            pops += (new_pop,)
            self._logger.info("Finished parsing {}".format(pop))
        return pops

    @abstractmethod
    def initialize_populations(self):
        return None, None

    @abstractmethod
    def train(self, n_iterations, stop_event=None):
        pass

    def log_results(self, batch_size, iteration, input_var, loader, **kwargs):
        append = ', '.join(['{}={}'.format(key, value) for key, value in kwargs.items()])
        if append:
            append = ', ' + append

        self._logger.info("Iteration={}, f(Generator(x))={}, f(Discriminator(x))={}{}".
                          format(iteration + 1, float(self.population_gen.individuals[0].fitness),
                                 float(self.population_dis.individuals[0].fitness),
                                 append))

        # Save max. 128 images unless configured otherwise
        sample_count = self.cc.settings['dataloader'].get('sample_count', False)
        batch_size = sample_count if sample_count else min(batch_size, 128)

        image_format = self.cc.settings['general']['logging']['image_format']

        path_real = os.path.join(self.cc.output_dir, 'real_images.{}'.format(image_format))
        path_fake = os.path.join(self.cc.output_dir, 'fake_images-{}.{}'.format(iteration + 1, image_format))

        self.save_images(batch_size, input_var, iteration, loader, path_fake, path_real)

        return path_real, path_fake

    def save_images(self, batch_size, input_var, iteration, loader, path_fake, path_real=None):
        # Check if dataset contains its own image conversion method (e.g. for gaussian values)
        if hasattr(loader.dataset, 'save_images'):
            # Save real images once
            if iteration == 0 and path_real:
                loader.dataset.save_images(input_var, path_real)

            z = noise(batch_size, self.network_factory.gen_input_size)
            if self.cc.settings['dataloader']['dataset_name'] == 'network_traffic':
                sequence_length = input_var.size(1)
                z = z.unsqueeze(1).repeat(1,sequence_length,1)

            if self.cc.settings['general']['logging'].get('print_multiple_generators', False):
                generated_output = []
                for i in range(min(len(self.population_gen.individuals), 5)):
                    gen = self.population_gen.individuals[i].genome.net
                    gen.eval()
                    generated_output.append(gen(z))
                    gen.train()
            else:
                gen = self.population_gen.individuals[0].genome.net
                gen.eval()
                generated_output = gen(z)
                gen.train()

            print_discriminator = self.cc.settings['general']['logging'].get('print_discriminator', False)
            discr = self.population_dis.individuals[0].genome if print_discriminator else None
            loader.dataset.save_images(generated_output, path_fake, discr)
        else:
            # Some datesets (e.g. ImageFolder) do not need shapes
            shape = loader.dataset.train_data.shape if hasattr(loader.dataset, 'train_data') else None

            # Save real images once
            if iteration == 0 and path_real:
                self.dataloader.save_images(input_var, shape, path_real)

            z = noise(batch_size, self.network_factory.gen_input_size)
            gen = self.population_gen.individuals[0].genome.net
            gen.eval()
            generated_output = gen(z)
            self.dataloader.save_images(generated_output, shape, path_fake)
            gen.train()

    def get_checkpoint_dir(self):
        output_base_dir = self.cc.output_dir.split('/distributed')[0]
        directory = os.path.join(output_base_dir, 'checkpoints')
        return directory

    def get_source_index(self, source):
        client_nodes = self.cc.settings['general']['distribution']['client_nodes']
        split_source = source.split(':')
        match = dict()
        match['address'] = split_source[0]
        match['port'] = int(split_source[1])
        match['id'] = source
        return client_nodes.index(match)

    def save_checkpoint(self, generators, gen_fitness, discriminators, disc_fitness, cell_number, grid_position, iteration):

        self._logger.info("Saving checkpoint")

        def get_individuals_information(individuals, prefix, cell_number, fitness):
            individuals_info = dict()
            self._logger.info("Individuals length: {}".format(len(individuals)))

            if len(individuals) > 0:
                individuals_info['learning_rate'] = '{}'.format(individuals[0].learning_rate)
                individuals_info['default_fitness'] = fitness
                individuals_info['individuals'] = []

                for individual in individuals:
                    indiv = dict()
                    indiv['id'] = individual.id
                    indiv['is_local'] = individual.is_local
                    indiv['fitness'] = individual.fitness
                    indiv['parameters'] = individual.genome.encoded_parameters
                    indiv['state_encoder'] = StateEncoder.encode(individual.optimizer_state)
                    # The individual.source parameter stores the network source of that individual represented by
                    # <ip addres>:<port>
                    indiv['source'] = individual.source
                    individuals_info['individuals'].append(indiv)

                    if indiv['is_local']:
                        filename = '{}{}.pkl'.format(prefix, cell_number)
                        torch.save(individual.genome.net.state_dict(),
                                   os.path.join(self.cc.output_dir, filename))
            return individuals_info

        checkpoint = dict()
        checkpoint['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        checkpoint['position'] = dict()
        checkpoint['position']['x'] = grid_position[0]
        checkpoint['position']['y'] = grid_position[1]
        checkpoint['iteration'] = iteration
        checkpoint['generators'] = get_individuals_information(generators, GENERATOR_PREFIX, cell_number, gen_fitness)
        checkpoint['discriminators'] = get_individuals_information(discriminators, DISCRIMINATOR_PREFIX, cell_number, disc_fitness)
        self._logger.info("Recording iter: {}".format(iteration))

        
        checkpoint_dir = self.get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        path_checkpoint = os.path.join(checkpoint_dir, 'checkpoint-{}.yml'.format(cell_number))
        with gzip.open(path_checkpoint, 'wt', encoding='UTF-8') as zipfile:
            yaml.dump(checkpoint, zipfile)

        self._logger.info("Finished saving checkpoint")
