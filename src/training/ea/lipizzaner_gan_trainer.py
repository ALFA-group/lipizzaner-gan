import random
from time import time

import numpy as np
import torch

from distribution.concurrent_populations import ConcurrentPopulations
from distribution.neighbourhood import Neighbourhood
from helpers.configuration_container import ConfigurationContainer
from helpers.db_logger import DbLogger
from helpers.population import TYPE_GENERATOR, TYPE_DISCRIMINATOR
from helpers.pytorch_helpers import to_pytorch_variable
from training.ea.ea_trainer import EvolutionaryAlgorithmTrainer
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset
from training.mixture.score_factory import ScoreCalculatorFactory


class LipizzanerGANTrainer(EvolutionaryAlgorithmTrainer):
    """
    Distributed, asynchronous trainer for coevolutionary GANs. Uses the standard Goodfellow GAN approach.
    """

    def __init__(self, dataloader, network_factory, population_size=10, tournament_size=2, mutation_probability=0.9,
                 n_replacements=1, sigma=0.25, alpha=0.25, default_adam_learning_rate=0.001, calc_mixture=False,
                 mixture_sigma=0.01, score_sample_size=10000, discriminator_skip_each_nth_step=0,
                 enable_selection=True):

        super().__init__(dataloader, network_factory, population_size, tournament_size, mutation_probability,
                         n_replacements, sigma, alpha)

        self.batch_number = 0

        self._default_adam_learning_rate = self.settings.get('default_adam_learning_rate', default_adam_learning_rate)
        self._discriminator_skip_each_nth_step = self.settings.get('discriminator_skip_each_nth_step',
                                                                   discriminator_skip_each_nth_step)
        self._enable_selection = self.settings.get('enable_selection', enable_selection)
        self.mixture_sigma = self.settings.get('mixture_sigma', mixture_sigma)

        self.neighbourhood = Neighbourhood.instance()

        for i, individual in enumerate(self.population_gen.individuals):
            individual.learning_rate = self._default_adam_learning_rate
            individual.id = '{}/G{}'.format(self.neighbourhood.cell_number, i)
        for i, individual in enumerate(self.population_dis.individuals):
            individual.learning_rate = self._default_adam_learning_rate
            individual.id = '{}/D{}'.format(self.neighbourhood.cell_number, i)

        self.concurrent_populations = ConcurrentPopulations.instance()
        self.concurrent_populations.generator = self.population_gen
        self.concurrent_populations.discriminator = self.population_dis
        self.concurrent_populations.unlock()

        experiment_id = ConfigurationContainer.instance().settings['general']['logging'].get('experiment_id', None)
        self.db_logger = DbLogger(current_experiment=experiment_id)

        if 'score' in self.settings and self.settings['score'].get('enabled', calc_mixture):
            self.score_calc = ScoreCalculatorFactory.create()
            self.score_sample_size = self.settings['score'].get('sample_size', score_sample_size)
            self.score = float('inf') if self.score_calc.is_reversed else float('-inf')
        else:
            self.score_calc = None
            self.score = 0

    def train(self, n_iterations, stop_event=None):

        loaded = self.dataloader.load()

        for iteration in range(n_iterations):
            self._logger.debug('Iteration {} started'.format(iteration))
            start_time = time()

            all_generators = self.neighbourhood.all_generators
            all_discriminators = self.neighbourhood.all_discriminators
            local_generators = self.neighbourhood.local_generators
            local_discriminators = self.neighbourhood.local_discriminators

            new_populations = {}

            self.batch_number = 0
            data_iterator = iter(loaded)
            while self.batch_number < len(loaded):
                input_data = next(data_iterator)[0]
                batch_size = input_data.size(0)
                input_data = to_pytorch_variable(self.dataloader.transpose_data(input_data))

                if iteration == 0 and self.batch_number == 0:
                    self._logger.debug('Evaluating first fitness')
                    self.evaluate_fitness(local_generators, all_discriminators, input_data)
                    self._logger.debug('Finished evaluating first fitness')

                if self.batch_number == 0 and self._enable_selection:
                    self._logger.debug('Started tournamend selection')
                    new_populations[TYPE_GENERATOR] = self.tournament_selection(all_generators, TYPE_GENERATOR)
                    new_populations[TYPE_DISCRIMINATOR] = self.tournament_selection(all_discriminators,
                                                                                    TYPE_DISCRIMINATOR)
                    self._logger.debug('Finished tournamend selection')

                # Quit if requested
                if stop_event is not None and stop_event.is_set():
                    self._logger.warning('External stop requested.')
                    return self.result()

                attackers = new_populations[TYPE_GENERATOR] if self._enable_selection else local_generators
                defenders = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else all_discriminators
                input_data = self.step(local_generators, attackers, defenders, input_data, loaded, data_iterator,
                                       self.neighbourhood.mixture_weights_discriminators)

                if self._discriminator_skip_each_nth_step == 0 or self.batch_number % (
                        self._discriminator_skip_each_nth_step + 1) == 0:
                    attackers = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else local_discriminators
                    defenders = new_populations[TYPE_GENERATOR] if self._enable_selection else all_generators

                    input_data = self.step(local_discriminators, attackers, defenders, input_data, loaded,
                                           data_iterator, self.neighbourhood.mixture_weights_generators)

                self._logger.info('Batch {}/{} done'.format(self.batch_number, len(loaded)))

                # If n_batches is set to 0, all batches will be used
                if self.is_last_batch(self.batch_number):
                    break

                self.batch_number += 1

            # Mutate mixture weights
            weights_generators = self.neighbourhood.mixture_weights_generators
            weights_discriminators = self.neighbourhood.mixture_weights_discriminators
            generators = new_populations[TYPE_GENERATOR] if self._enable_selection else all_generators
            discriminators = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else all_discriminators
            self.mutate_mixture_weights(weights_generators, weights_discriminators, generators, discriminators,
                                        input_data)
            self.mutate_mixture_weights(weights_discriminators, weights_generators, discriminators, generators,
                                        input_data)

            # Replace the worst with the best new
            if self._enable_selection:
                self.evaluate_fitness(new_populations[TYPE_GENERATOR], new_populations[TYPE_DISCRIMINATOR], input_data)
                self.concurrent_populations.lock()
                local_generators.replacement(new_populations[TYPE_GENERATOR], self._n_replacements)
                local_generators.sort_population()
                local_discriminators.replacement(new_populations[TYPE_DISCRIMINATOR], self._n_replacements)
                local_discriminators.sort_population()
                self.concurrent_populations.unlock()
            else:
                self.evaluate_fitness(all_generators, all_discriminators, input_data)

            if self.score_calc is not None:
                self._logger.info('Calculating FID/inception score.')
                self.calculate_score()

            stop_time = time()

            path_real_images, path_fake_images = \
                self.log_results(batch_size, iteration, input_data, loaded,
                                 lr_gen=self.concurrent_populations.generator.individuals[0].learning_rate,
                                 lr_dis=self.concurrent_populations.discriminator.individuals[0].learning_rate,
                                 score=self.score, mixture_gen=self.neighbourhood.mixture_weights_generators,
                                 mixture_dis=self.neighbourhood.mixture_weights_discriminators)

            if self.db_logger.is_enabled:
                self.db_logger.log_results(iteration, self.neighbourhood, self.concurrent_populations,
                                           self.score, stop_time - start_time,
                                           path_real_images, path_fake_images)

        return self.result()

    def step(self, original, attacker, defender, input_data, loaded, data_iterator, defender_weights):
        # Don't execute for remote populations - needed if generator and discriminator are on different node
        if any(not ind.is_local for ind in original.individuals):
            return

        self.mutate_hyperparams(attacker)
        return self.update_genomes(attacker, defender, input_data, loaded, data_iterator, defender_weights)

    def is_last_batch(self, i):
        return self.dataloader.n_batches != 0 and self.dataloader.n_batches - 1 == i

    def result(self):
        return ((self.concurrent_populations.generator.individuals[0].genome,
                 self.concurrent_populations.generator.individuals[0].fitness),
                (self.concurrent_populations.discriminator.individuals[0].genome,
                 self.concurrent_populations.discriminator.individuals[0].fitness))

    def mutate_hyperparams(self, population):
        loc = -(self._default_adam_learning_rate / 10)
        deltas = np.random.normal(loc=loc, scale=self._default_adam_learning_rate, size=len(population.individuals))
        deltas[np.random.rand(*deltas.shape) < 1 - self._mutation_probability] = 0
        for i, individual in enumerate(population.individuals):
            individual.learning_rate = max(0, individual.learning_rate + deltas[i] * self._alpha)

    def update_genomes(self, population_attacker, population_defender, input_var, loaded, data_iterator,
                       defender_weights):

        for individual_attacker in population_attacker.individuals:
            attacker = individual_attacker.genome
            weights = [self.get_weight(defender, defender_weights) for defender in population_defender.individuals]
            weights /= np.sum(weights)
            defender = np.random.choice(population_defender.individuals, p=weights).genome
            optimizer = torch.optim.Adam(attacker.net.parameters(),
                                         lr=individual_attacker.learning_rate,
                                         betas=(0.5, 0.999))

            # Restore previous state dict, if available
            if individual_attacker.optimizer_state is not None:
                optimizer.load_state_dict(individual_attacker.optimizer_state)

            loss = attacker.compute_loss_against(defender, input_var)[0]

            attacker.net.zero_grad()
            defender.net.zero_grad()
            loss.backward()
            optimizer.step()

            individual_attacker.optimizer_state = optimizer.state_dict()

        return input_var

    @staticmethod
    def evaluate_fitness(population_attacker, population_defender, input_var):

        for individual_attacker in population_attacker.individuals:
            for individual_defender in population_defender.individuals:

                if individual_attacker.is_local:
                    fitness_attacker = float(individual_attacker.genome.compute_loss_against(
                        individual_defender.genome, input_var)[0])

                    if fitness_attacker > individual_attacker.fitness:
                        individual_attacker.fitness = fitness_attacker

                if individual_defender.is_local:
                    fitness_defender = float(individual_defender.genome.compute_loss_against(
                        individual_attacker.genome, input_var)[0])

                    if fitness_defender > individual_defender.fitness:
                        individual_defender.fitness = fitness_defender

    def mutate_mixture_weights(self, weights_attacker, weights_defender, population_attacker, population_defender,
                               input_data):

        # Not necessary for single-cell grids, as mixture must always be [1]
        if self.neighbourhood.grid_size == 1:
            return

        # Mutate mixture weights
        z = np.random.normal(loc=0, scale=self.mixture_sigma, size=len(weights_attacker))
        transformed = np.asarray([value for _, value in weights_attacker.items()])
        transformed += z

        new_mixture_weights = {}
        for i, key in enumerate(weights_attacker):
            new_mixture_weights[key] = transformed[i]

        for attacker in population_attacker.individuals:
            loss_prev = self.weights_loss(attacker, population_defender, weights_attacker, weights_defender, input_data)
            loss_new = self.weights_loss(attacker, population_defender, new_mixture_weights, weights_defender,
                                         input_data)

            if loss_new < loss_prev:
                weights_attacker[attacker.source] = self.get_weight(attacker, new_mixture_weights)

        # Don't allow negative values, normalize to sum of 1.0
        clipped = np.clip(list(weights_attacker.values()), 0, None)
        clipped /= np.sum(clipped)
        for i, key in enumerate(weights_attacker):
            weights_attacker[key] = clipped[i]

    def calculate_score(self):
        best_generators = self.neighbourhood.best_generators

        dataset = MixedGeneratorDataset(best_generators, self.neighbourhood.mixture_weights_generators,
                                        self.score_sample_size)
        self.score = self.score_calc.calculate(dataset)[0]

    @staticmethod
    def get_weight(individual, weights):
        return [v for k, v in weights.items() if k == individual.source][0]

    @staticmethod
    def weights_loss(attacker, population_defender, weights_attacker, weights_defender, input_data):
        w_attacker = LipizzanerGANTrainer.get_weight(attacker, weights_attacker)
        return sum([w_attacker * LipizzanerGANTrainer.get_weight(defender, weights_defender) *
                    float(attacker.genome.compute_loss_against(defender.genome, input_data)[0]) for defender in
                    population_defender.individuals])
