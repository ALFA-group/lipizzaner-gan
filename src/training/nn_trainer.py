import logging
import os
from abc import abstractmethod, ABC

from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import noise

import yaml
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

        self.population_gen, self.population_dis = self.initialize_populations()

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

    def save_checkpoint(self, generators, discriminators, cell_number, grid_position):

        def get_individuals_information(individuals, prefix, cell_number):
            individuals_info = dict()

            if len(individuals) > 0:
                individuals_info['iteration'] = individuals[0].iteration
                individuals_info['learning_rate'] = '{}'.format(individuals[0].learning_rate)
                individuals_info['individuals'] = []

                for individual in individuals:
                    indiv = dict()
                    indiv['id'] = individual.id
                    indiv['is_local'] = individual.is_local
                    indiv['fitness'] = individual.fitness
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
        checkpoint['generators'] = get_individuals_information(generators, GENERATOR_PREFIX, cell_number)
        checkpoint['discriminators'] = get_individuals_information(discriminators, DISCRIMINATOR_PREFIX, cell_number)

        path_checkpoint = os.path.join(self.cc.output_dir, 'checkpoint-{}.yml'.format(cell_number))
        with open(path_checkpoint, 'w') as file:
            yaml.dump(checkpoint, file)
