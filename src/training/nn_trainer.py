import logging
import os
from abc import abstractmethod, ABC

from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import noise


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
