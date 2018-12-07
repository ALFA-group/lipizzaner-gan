import random
from itertools import tee

import numpy as np
import torch
from torch.autograd import Variable

from helpers.population import TYPE_DISCRIMINATOR, TYPE_GENERATOR
from helpers.pytorch_helpers import is_cuda_enabled, noise, to_pytorch_variable
from training.ea.lipizzaner_gan_trainer import LipizzanerGANTrainer

DISCRIMINATOR_STEPS = 5
CLAMP_LOWER = -0.01
CLAMP_UPPER = 0.01


class LipizzanerWGANTrainer(LipizzanerGANTrainer):
    """
    Distributed, asynchronous trainer for coevolutionary GANs. Uses the more sophisticated Wasserstein GAN.
    Original source code: from https://github.com/martinarjovsky/WassersteinGAN/
    """

    gen_iterations = 0
    real_labels = torch.FloatTensor([1]).cuda() if is_cuda_enabled() else torch.FloatTensor([1])
    fake_labels = real_labels * -1

    def update_genomes(self, population_attacker, population_defender, input_var, loaded, data_iterator):
        if population_attacker.population_type == TYPE_DISCRIMINATOR:
            return self._update_discriminators(population_attacker, population_defender, input_var, loaded,
                                        data_iterator)
        elif population_attacker.population_type == TYPE_GENERATOR:
            return self._update_generators(population_attacker, population_defender, input_var)
        else:
            raise Exception('Population type not explicitely set.')

    def _update_discriminators(self, population_attacker, population_defender, input_var, loaded, data_iterator):

        batch_size = input_var.size(0)
        # Randomly pick one only, referred from asynchronous_ea_trainer
        generator = random.choice(population_defender.individuals)

        for i, discriminator in enumerate(population_attacker.individuals):
            if i < len(population_attacker.individuals) - 1:
                # https://stackoverflow.com/a/42132767
                # Perform deep copy first instead of directly updating iterator passed in
                data_iterator, curr_iterator = tee(data_iterator)
            else:
                # Directly update the iterator with the last individual only, so that
                # every individual can learn from the full batch
                curr_iterator = data_iterator

            # Use temporary batch variable for each individual
            # so that every individual can learn from the full batch
            curr_batch_number = self.batch_number
            optimizer = self._get_optimizer(discriminator)

            # Train the discriminator Diters times
            if self.gen_iterations < 25 or self.gen_iterations % 500 == 0:
                discriminator_iterations = 100
            else:
                discriminator_iterations = DISCRIMINATOR_STEPS

            j = 0
            while j < discriminator_iterations and curr_batch_number < len(loaded):
                if j > 0:
                    input_var = to_pytorch_variable(self.dataloader.transpose_data(next(curr_iterator)[0]))
                j += 1

                # Train with real data
                discriminator.genome.net.zero_grad()
                error_real = discriminator.genome.net(input_var)
                error_real = error_real.mean(0).view(1)
                error_real.backward(self.real_labels)

                # Train with fake data
                z = noise(batch_size, generator.genome.data_size)
                z.volatile = True
                fake_data = Variable(generator.genome.net(z).data)
                loss = discriminator.genome.net(fake_data).mean(0).view(1)
                loss.backward(self.fake_labels)
                optimizer.step()

                # Clamp parameters to a cube
                for p in discriminator.genome.net.parameters():
                    p.data.clamp_(CLAMP_LOWER, CLAMP_UPPER)

                curr_batch_number += 1

            discriminator.optimizer_state = optimizer.state_dict()
        # Update the final batch_number to class variable after all individuals are updated
        self.batch_number = curr_batch_number

        return input_var


    def _update_generators(self, population_attacker, population_defender, input_var):

        batch_size = input_var.size(0)
        # Randomly pick one only, referred from asynchronous_ea_trainer
        discriminator = random.choice(population_defender.individuals)

        for generator in population_attacker.individuals:
            optimizer = self._get_optimizer(generator)

            # Avoid computation
            for p in discriminator.genome.net.parameters():
                p.requires_grad = False

            generator.genome.net.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            z = noise(batch_size, generator.genome.data_size)
            fake_data = generator.genome.net(z)
            error = discriminator.genome.net(fake_data).mean(0).view(1)
            error.backward(self.real_labels)
            optimizer.step()

            generator.optimizer_state = optimizer.state_dict()

            for p in discriminator.genome.net.parameters():
                p.requires_grad = True

        self.gen_iterations += 1
        return input_var

    @staticmethod
    def _get_optimizer(individual):
        optimizer = torch.optim.RMSprop(individual.genome.net.parameters(),
                                        lr=individual.learning_rate)

        # Restore previous state dict, if available
        if individual.optimizer_state is not None:
            optimizer.load_state_dict(individual.optimizer_state)
        return optimizer
