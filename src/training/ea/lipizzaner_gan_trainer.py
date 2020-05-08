import random
from time import time
from collections import OrderedDict

import numpy as np
import torch

from distribution.concurrent_populations import ConcurrentPopulations
from distribution.neighbourhood import Neighbourhood
from helpers.configuration_container import ConfigurationContainer
from helpers.db_logger import DbLogger
from helpers.population import TYPE_GENERATOR, TYPE_DISCRIMINATOR
from helpers.pytorch_helpers import to_pytorch_variable
from helpers.pytorch_helpers import noise
from training.ea.ea_trainer import EvolutionaryAlgorithmTrainer
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset
from training.mixture.score_factory import ScoreCalculatorFactory

from data.network_data_loader import generate_random_sequences


class LipizzanerGANTrainer(EvolutionaryAlgorithmTrainer):
    """
    Distributed, asynchronous trainer for coevolutionary GANs. Uses the standard Goodfellow GAN approach.
    (Without discriminator mixture)
    """

    def __init__(self, dataloader, network_factory, population_size=10, tournament_size=2, mutation_probability=0.9,
                 n_replacements=1, sigma=0.25, alpha=0.25, default_adam_learning_rate=0.001, calc_mixture=False,
                 mixture_sigma=0.01, score_sample_size=10000, discriminator_skip_each_nth_step=0,
                 enable_selection=True, fitness_sample_size=10000, calculate_net_weights_dist=False,
                 fitness_mode='worst',  es_generations=10, es_score_sample_size=10000, es_random_init=False,
                 checkpoint_period=0):

        super().__init__(dataloader, network_factory, population_size, tournament_size, mutation_probability,
                         n_replacements, sigma, alpha)

        self.batch_number = 0
        self.cc = ConfigurationContainer.instance()

        self._default_adam_learning_rate = self.settings.get('default_adam_learning_rate', default_adam_learning_rate)
        self._discriminator_skip_each_nth_step = self.settings.get('discriminator_skip_each_nth_step',
                                                                   discriminator_skip_each_nth_step)
        self._enable_selection = self.settings.get('enable_selection', enable_selection)
        self.mixture_sigma = self.settings.get('mixture_sigma', mixture_sigma)

        self.neighbourhood = Neighbourhood.instance()

        for i, individual in enumerate(self.population_gen.individuals):
            individual.learning_rate = self.settings.get(
                'default_g_adam_learning_rate',
                self._default_adam_learning_rate
            )
            individual.id = '{}/G{}'.format(self.neighbourhood.cell_number, i)
        for i, individual in enumerate(self.population_dis.individuals):
            individual.learning_rate = self.settings.get(
                'default_d_adam_learning_rate',
                self._default_adam_learning_rate
            )
            individual.id = '{}/D{}'.format(self.neighbourhood.cell_number, i)

        self.concurrent_populations = ConcurrentPopulations.instance()
        self.concurrent_populations.generator = self.population_gen
        self.concurrent_populations.discriminator = self.population_dis
        self.concurrent_populations.unlock()

        experiment_id = self.cc.settings['general']['logging'].get('experiment_id', None)
        self.db_logger = DbLogger(current_experiment=experiment_id)

        if 'fitness' in self.settings:
            self.fitness_sample_size = self.settings['fitness'].get('fitness_sample_size', fitness_sample_size)
            self.fitness_loaded = self.dataloader.load()
            self.fitness_iterator = iter(self.fitness_loaded)  # Create iterator for fitness loader

            # Determine how to aggregate fitness calculated among neighbourhood
            self.fitness_mode = self.settings['fitness'].get('fitness_mode', fitness_mode)
            if self.fitness_mode not in ['worse', 'best', 'average']:
                raise NotImplementedError("Invalid argument for fitness_mode: {}".format(self.fitness_mode))
        else:
            # TODO: Add code for safe implementation & error handling
            raise KeyError("Fitness section must be defined in configuration file")

        if 'score' in self.settings and self.settings['score'].get('enabled', calc_mixture):
            self.score_calc = ScoreCalculatorFactory.create()
            self.score_sample_size = self.settings['score'].get('sample_size', score_sample_size)
            self.score = float('inf') if self.score_calc.is_reversed else float('-inf')
            self.mixture_generator_samples_mode = self.cc.settings['trainer']['mixture_generator_samples_mode']
        elif 'optimize_mixture' in self.settings:
            self.score_calc = ScoreCalculatorFactory.create()
            self.score = float('inf') if self.score_calc.is_reversed else float('-inf')
        else:
            self.score_sample_size = score_sample_size
            self.score_calc = None
            self.score = 0

        if 'optimize_mixture' in self.settings:
            self.optimize_weights_at_the_end = True
            self.score_sample_size = self.settings['optimize_mixture'].get('sample_size', es_score_sample_size)
            self.es_generations = self.settings['optimize_mixture'].get('es_generations', es_generations)
            self.es_random_init = self.settings['optimize_mixture'].get('es_random_init', es_random_init)
            self.mixture_sigma = self.settings['optimize_mixture'].get('mixture_sigma', mixture_sigma)
            self.mixture_generator_samples_mode = self.cc.settings['trainer']['mixture_generator_samples_mode']
        else:
            self.optimize_weights_at_the_end = False

        n_iterations = self.cc.settings['trainer'].get('n_iterations', 0)
        assert 0 <= checkpoint_period <= n_iterations, 'Checkpoint period paramenter (checkpoint_period) should be ' \
                                                       'between 0 and the number of iterations (n_iterations).'
        self.checkpoint_period = self.cc.settings['general'].get('checkpoint_period', checkpoint_period)



    def train(self, n_iterations, stop_event=None):
        loaded = self.dataloader.load()


        for iteration in range(n_iterations):
            self._logger.debug('Iteration {} started'.format(iteration + 1))
            start_time = time()

            all_generators = self.neighbourhood.all_generators
            all_discriminators = self.neighbourhood.all_discriminators
            local_generators = self.neighbourhood.local_generators
            local_discriminators = self.neighbourhood.local_discriminators

            alpha = self.neighbourhood.alpha
            beta = self.neighbourhood.beta
            if alpha is not None:
                self._logger.info(f'Alpha is {alpha} and Beta is {beta}')
            else:
                self._logger.debug('Alpha and Beta are not set')

            # Log the name of individuals in entire neighborhood and local individuals for every iteration
            # (to help tracing because individuals from adjacent cells might be from different iterations)
            self._logger.info('Neighborhood located in possition {} of the grid'.format(self.neighbourhood.grid_position))
            self._logger.info('Generators in current neighborhood are {}'.format([
                individual.name for individual in all_generators.individuals
            ]))
            self._logger.info('Discriminators in current neighborhood are {}'.format([
                individual.name for individual in all_discriminators.individuals
            ]))
            self._logger.info('Local generators in current neighborhood are {}'.format([
                individual.name for individual in local_generators.individuals
            ]))
            self._logger.info('Local discriminators in current neighborhood are {}'.format([
                individual.name for individual in local_discriminators.individuals
            ]))

            self._logger.info('L2 distance between all generators weights: {}'.format(all_generators.net_weights_dist))
            self._logger.info(
                'L2 distance between all discriminators weights: {}'.format(all_discriminators.net_weights_dist))

            new_populations = {}

            # Create random dataset to evaluate fitness in each iterations
            fitness_input, fitness_labels = self.generate_random_fitness_samples(self.fitness_sample_size)
            if self.cc.settings['dataloader']['dataset_name'] == 'celeba' \
                    or self.cc.settings['dataloader']['dataset_name'] == 'cifar' \
                    or self.cc.settings['network']['name'] == 'ssgan_convolutional_mnist':
                fitness_input = to_pytorch_variable(fitness_input)
                fitness_labels = to_pytorch_variable(fitness_labels)
            elif self.cc.settings['dataloader']['dataset_name'] == 'network_traffic':
                fitness_input = to_pytorch_variable(generate_random_sequences(self.fitness_sample_size))
            else:
                fitness_input = to_pytorch_variable(fitness_input.view(self.fitness_sample_size, -1))
                fitness_labels = to_pytorch_variable(fitness_labels.view(self.fitness_sample_size, -1))

            fitness_labels = torch.squeeze(fitness_labels)

            # Fitness evaluation
            self._logger.debug('Evaluating fitness')
            self.evaluate_fitness(all_generators, all_discriminators, fitness_input, self.fitness_mode)
            self.evaluate_fitness(all_discriminators, all_generators, fitness_input,
                                  self.fitness_mode, labels=fitness_labels,
                                  logger=self._logger, alpha=alpha, beta=beta,
                                  iter=iteration, log_class_distribution=True)
            self._logger.debug('Finished evaluating fitness')

            # Tournament selection
            if self._enable_selection:
                self._logger.debug('Started tournament selection')
                new_populations[TYPE_GENERATOR] = self.tournament_selection(all_generators,
                                                                            TYPE_GENERATOR,
                                                                            is_logging=True)
                new_populations[TYPE_DISCRIMINATOR] = self.tournament_selection(all_discriminators,
                                                                                TYPE_DISCRIMINATOR,
                                                                                is_logging=True)
                self._logger.debug('Finished tournament selection')

            self.batch_number = 0
            data_iterator = iter(loaded)
            while self.batch_number < len(loaded):
                if self.cc.settings['dataloader']['dataset_name'] == 'network_traffic':
                    input_data = to_pytorch_variable(next(data_iterator))
                    batch_size = input_data.size(0)
                else:
                    input_data, labels = next(data_iterator)
                    batch_size = input_data.size(0)
                    input_data = to_pytorch_variable(self.dataloader.transpose_data(input_data))
                    labels = to_pytorch_variable(self.dataloader.transpose_data(labels))
                    labels = torch.squeeze(labels)

                # Quit if requested
                if stop_event is not None and stop_event.is_set():
                    self._logger.warning('External stop requested.')
                    return self.result()

                attackers = new_populations[TYPE_GENERATOR] if self._enable_selection else local_generators
                defenders = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else all_discriminators
                input_data = self.step(local_generators, attackers, defenders, input_data, self.batch_number, loaded,
                                       data_iterator, iter=iteration)

                if self._discriminator_skip_each_nth_step == 0 or self.batch_number % (
                        self._discriminator_skip_each_nth_step + 1) == 0:
                    self._logger.debug('Skipping discriminator step')

                    attackers = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else local_discriminators
                    defenders = new_populations[TYPE_GENERATOR] if self._enable_selection else all_generators
                    input_data = self.step(local_discriminators, attackers, defenders, input_data, self.batch_number,
                                           loaded, data_iterator, labels=labels, alpha=alpha, beta=beta, iter=iteration)

                self._logger.info('Iteration {}, Batch {}/{}'.format(iteration + 1, self.batch_number, len(loaded)))

                # If n_batches is set to 0, all batches will be used
                if self.is_last_batch(self.batch_number):
                    break

                self.batch_number += 1

            # Perform selection first before mutation of mixture_weights
            # Replace the worst with the best new
            if self._enable_selection:
                # Evaluate fitness of new_populations against neighborhood
                self.evaluate_fitness(new_populations[TYPE_GENERATOR], all_discriminators, fitness_input,
                                      self.fitness_mode)
                self.evaluate_fitness(new_populations[TYPE_DISCRIMINATOR], all_generators, fitness_input,
                                      self.fitness_mode, labels=fitness_labels, alpha=alpha, beta=beta, iter=iteration)
                self.concurrent_populations.lock()
                local_generators.replacement(new_populations[TYPE_GENERATOR], self._n_replacements, is_logging=True)
                local_generators.sort_population(is_logging=True)
                local_discriminators.replacement(new_populations[TYPE_DISCRIMINATOR], self._n_replacements,
                                                 is_logging=True)
                local_discriminators.sort_population(is_logging=True)
                self.concurrent_populations.unlock()

                # Update individuals' iteration and id after replacement and logging to ease tracing
                for i, individual in enumerate(local_generators.individuals):
                    individual.id = '{}/G{}'.format(self.neighbourhood.cell_number, i)
                    individual.iteration = iteration + 1
                for i, individual in enumerate(local_discriminators.individuals):
                    individual.id = '{}/D{}'.format(self.neighbourhood.cell_number, i)
                    individual.iteration = iteration + 1
            else:
                # Re-evaluate fitness of local_generators and local_discriminators against neighborhood
                self.evaluate_fitness(local_generators, all_discriminators, fitness_input, self.fitness_mode)
                self.evaluate_fitness(local_discriminators, all_generators,
                                      fitness_input, self.fitness_mode,
                                      labels=fitness_labels, alpha=alpha, beta=beta, iter=iteration)


            # Mutate mixture weights after selection
            if not self.optimize_weights_at_the_end:
                self.mutate_mixture_weights_with_score(input_data)  # self.score is updated here

            stop_time = time()

            path_real_images, path_fake_images = \
                self.log_results(batch_size, iteration, input_data, loaded,
                                 lr_gen=self.concurrent_populations.generator.individuals[0].learning_rate,
                                 lr_dis=self.concurrent_populations.discriminator.individuals[0].learning_rate,
                                 score=self.score, mixture_gen=self.neighbourhood.mixture_weights_generators,
                                 mixture_dis=None)

            if self.db_logger.is_enabled:
                self.db_logger.log_results(iteration, self.neighbourhood, self.concurrent_populations,
                                           self.score, stop_time - start_time,
                                           path_real_images, path_fake_images)

            if self.checkpoint_period>0 and (iteration+1)%self.checkpoint_period==0:
                self.save_checkpoint(all_generators.individuals, all_discriminators.individuals,
                                     self.neighbourhood.cell_number, self.neighbourhood.grid_position)


        discriminator = self.concurrent_populations.discriminator.individuals[0].genome

        if 'ssgan' in self.cc.settings['network']['name']:
            batch_size = self.dataloader.batch_size
            dataloader_loaded = self.dataloader.load(train=True)
            self.test(discriminator, dataloader_loaded, train=True)

            self.dataloader.batch_size = 100
            dataloader_loaded = self.dataloader.load(train=False)
            self.test(discriminator, dataloader_loaded, train=False)

            discriminators = [individual.genome for individual in self.neighbourhood.all_discriminators.individuals]
            dataloader_loaded = self.dataloader.load(train=False)
            self.test_majority_voting(discriminators, dataloader_loaded, train=False)
            self.dataloader.batch_size = batch_size


        if self.optimize_weights_at_the_end:
            self.optimize_generator_mixture_weights()

            path_real_images, path_fake_images = \
                self.log_results(batch_size, iteration+1, input_data, loaded,
                                 lr_gen=self.concurrent_populations.generator.individuals[0].learning_rate,
                                 lr_dis=self.concurrent_populations.discriminator.individuals[0].learning_rate,
                                 score=self.score, mixture_gen=self.neighbourhood.mixture_weights_generators,
                                 mixture_dis=self.neighbourhood.mixture_weights_discriminators)

            if self.db_logger.is_enabled:
                self.db_logger.log_results(iteration+1, self.neighbourhood, self.concurrent_populations,
                                           self.score, stop_time - start_time,
                                           path_real_images, path_fake_images)


        return self.result()

    def test_majority_voting(self, models, test_loader, train=False):
        correct = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_or_test = 'Train' if train else 'Test'
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 784)
                # data = data.view(-1, 1, 28, 28)
                pred_accumulator = []
                for model in models:
                    output = model.classification_layer(model.net(data))
                    output = output.view(-1, 11)
                    pred = output.argmax(dim=1, keepdim=True)
                    pred_accumulator.append(pred.view(-1))
                label_votes = torch.tensor(list(zip(*pred_accumulator)))
                prediction = torch.tensor([labels.bincount(minlength=11).argmax() for labels in label_votes])
                print(prediction.shape)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        num_samples = len(test_loader.dataset)
        accuracy = 100.0 * float(correct / num_samples)
        self._logger.info(f'Majority Voting {train_or_test} Accuracy: {correct}/{num_samples} ({accuracy}%)')

    def test(self, model, test_loader, train=False):
        correct = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_or_test = 'Train' if train else 'Test'
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 784)
                # data = data.view(-1, 1, 28, 28)
                output = model.classification_layer(model.net(data))
                output = output.view(-1, 11)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        num_samples = len(test_loader.dataset)
        accuracy = 100.0 * float(correct / num_samples)
        self._logger.info(f'{train_or_test} Accuracy: {correct}/{num_samples} ({accuracy}%)')

    def optimize_generator_mixture_weights(self):
        generators = self.neighbourhood.best_generators
        weights_generators = self.neighbourhood.mixture_weights_generators

        # Not necessary for single-cell grids, as mixture must always be [1]
        if self.neighbourhood.grid_size == 1:
            return

        # Create random vector from latent space
        z_noise = noise(self.score_sample_size, generators.individuals[0].genome.data_size)

        # Include option to start from random weights
        if self.es_random_init:
            aux_weights = np.random.rand(len(weights_generators))
            aux_weights /= np.sum(aux_weights)
            weights_generators = OrderedDict(zip(weights_generators.keys(), aux_weights))
            self.neighbourhood.mixture_weights_generators = weights_generators

        dataset = MixedGeneratorDataset(generators, weights_generators,
                                          self.score_sample_size, self.mixture_generator_samples_mode, z_noise)

        self.score = self.score_calc.calculate(dataset)[0]
        init_score = self.score

        self._logger.info(
            'Mixture weight mutation - Starting mixture weights optimization ...')
        self._logger.info('Init score: {}\tInit weights: {}.'.format(init_score, weights_generators))

        for g in range(self.es_generations):

            # Mutate mixture weights
            z = np.random.normal(loc=0, scale=self.mixture_sigma, size=len(weights_generators))
            transformed = np.asarray([value for _, value in weights_generators.items()])
            transformed += z

            # Don't allow negative values, normalize to sum of 1.0
            transformed = np.clip(transformed, 0, None)
            transformed /= np.sum(transformed)
            new_mixture_weights = OrderedDict(zip(weights_generators.keys(), transformed))

            # TODO: Testing the idea of not generating the images again
            dataset = MixedGeneratorDataset(generators, new_mixture_weights,
                                              self.score_sample_size,
                                              self.mixture_generator_samples_mode, z_noise)

            if self.score_calc is not None:
                score_after_mutation = self.score_calc.calculate(dataset)[0]
                self._logger.info(
                    'Mixture weight mutation - Generation: {} \tScore of new weights: {}\tNew weights: {}.'.format(g,
                                                                                               score_after_mutation,
                                                                                               new_mixture_weights))

                # For fid the lower the better, for inception_score, the higher the better
                if (score_after_mutation < self.score and self.score_calc.is_reversed) \
                        or (score_after_mutation > self.score and (not self.score_calc.is_reversed)):
                    weights_generators = new_mixture_weights
                    self.score = score_after_mutation
                    self._logger.info(
                        'Mixture weight mutation - Generation: {} \tNew score: {}\tWeights changed to: {}.'.format(g,
                                                                                                    self.score,
                                                                                                    weights_generators))
        self.neighbourhood.mixture_weights_generators = weights_generators

        self._logger.info(
            'Mixture weight mutation - Score before mixture weight optimzation: {}\tScore after mixture weight optimzation: {}.'.format(
                                                                                                        init_score,
                                                                                                        self.score))

    def step(self, original, attacker, defender, input_data, i, loaded,
             data_iterator, labels=None, alpha=None, beta=None, iter=None):
        self.mutate_hyperparams(attacker)
        return self.update_genomes(attacker, defender, input_data, loaded,
                                   data_iterator, labels=labels, alpha=alpha, beta=beta, iter=iter)

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

    def update_genomes(self, population_attacker, population_defender, input_var,
                       loaded, data_iterator, labels=None, alpha=None, beta=None, iter=None):

        # TODO Currently picking random opponent, introduce parameter for this
        defender = random.choice(population_defender.individuals).genome

        for individual_attacker in population_attacker.individuals:
            attacker = individual_attacker.genome
            optimizer = torch.optim.Adam(attacker.net.parameters(),
                                         lr=individual_attacker.learning_rate,
                                         betas=(0.5, 0.999))

            # Restore previous state dict, if available
            if individual_attacker.optimizer_state is not None:
                optimizer.load_state_dict(individual_attacker.optimizer_state)

            if labels is None:
                loss = attacker.compute_loss_against(defender, input_var)[0]
            else:
                loss = attacker.compute_loss_against(defender, input_var,
                                                     labels=labels, alpha=alpha, beta=beta, iter=iter)[0]

            attacker.net.zero_grad()
            defender.net.zero_grad()
            loss.backward()
            optimizer.step()

            individual_attacker.optimizer_state = optimizer.state_dict()

        return input_var

    @staticmethod
    def evaluate_fitness(population_attacker, population_defender, input_var,
                         fitness_mode, labels=None, logger=None, alpha=None,
                         beta=None, iter=None, log_class_distribution=False):
        # Single direction only: Evaluate fitness of attacker based on defender
        # TODO: Simplify and refactor this function
        def compare_fitness(curr_fitness, fitness, mode):
            # The initial fitness value is -inf before evaluation started, so we
            # directly adopt the curr_fitness when -inf is encountered
            if mode == 'best':
                if curr_fitness < fitness or fitness == float('-inf'):
                    return curr_fitness
            elif mode == 'worse':
                if curr_fitness > fitness or fitness == float('-inf'):
                    return curr_fitness
            elif mode == 'average':
                if fitness == float('-inf'):
                    return curr_fitness
                else:
                    return fitness + curr_fitness

            return fitness

        for individual_attacker in population_attacker.individuals:
            individual_attacker.fitness = float(
                '-inf')  # Reinitalize before evaluation started (Needed for average fitness)
            for individual_defender in population_defender.individuals:
                if labels is None:
                    fitness_attacker = float(
                        individual_attacker.genome.compute_loss_against(
                            individual_defender.genome, input_var)[0])
                else:
                    fitness_attacker = float(
                        individual_attacker.genome.compute_loss_against(
                            individual_defender.genome, input_var,
                            labels=labels, alpha=alpha, beta=beta, iter=iter)[0])

                individual_attacker.fitness = compare_fitness(fitness_attacker, individual_attacker.fitness,
                                                              fitness_mode)

            if fitness_mode == 'average':
                individual_attacker.fitness /= len(population_defender.individuals)

        if labels is not None and logger is not None:
            gen = None
            for g in population_defender.individuals:
                if g.is_local:
                    gen = g
                    break
            dis = None
            for d in population_attacker.individuals:
                if d.is_local:
                    dis = d
                    break
            generator = gen.genome
            discriminator = dis.genome
            discriminator_output = discriminator.compute_loss_against(
                generator, input_var, labels=labels, alpha=alpha, beta=beta,
                iter=iter, log_class_distribution=log_class_distribution
            )
            accuracy = discriminator_output[2]
            if discriminator.name == "SemiSupervisedDiscriminator" and \
                    accuracy is not None:
                # d_acc = accuracy[0]
                # real_acc = accuracy[1]
                # fake_acc = accuracy[2]
                logger.info(
                    f"Iteration {iter},  Label Prediction Accuracy: {100 * accuracy}% "
                    # "Real Image Prediction Accuracy: %d%%, "
                    # "Fake Image Prediction Accuracy: %d%%"
                    # % (100 * d_acc, 100 * real_acc, 100 * (1 - fake_acc))
                )


    def mutate_mixture_weights_with_score(self, input_data):
        # Not necessary for single-cell grids, as mixture must always be [1]
        if self.neighbourhood.grid_size == 1:
            if self.score_calc is not None:
                self._logger.info('Calculating FID/inception score.')
                best_generators = self.neighbourhood.best_generators

                dataset = MixedGeneratorDataset(best_generators,
                                                self.neighbourhood.mixture_weights_generators,
                                                self.score_sample_size,
                                                self.cc.settings['trainer']['mixture_generator_samples_mode'])
                self.score = self.score_calc.calculate(dataset)[0]
        else:
            # Mutate mixture weights
            z = np.random.normal(loc=0, scale=self.mixture_sigma,
                                 size=len(self.neighbourhood.mixture_weights_generators))
            transformed = np.asarray([value for _, value in self.neighbourhood.mixture_weights_generators.items()])
            transformed += z
            # Don't allow negative values, normalize to sum of 1.0
            transformed = np.clip(transformed, 0, None)
            transformed /= np.sum(transformed)

            new_mixture_weights_generators = OrderedDict(
                zip(self.neighbourhood.mixture_weights_generators.keys(), transformed))

            best_generators = self.neighbourhood.best_generators
            dataset_before_mutation = MixedGeneratorDataset(best_generators,
                                                            self.neighbourhood.mixture_weights_generators,
                                                            self.score_sample_size,
                                                            self.cc.settings['trainer'][
                                                                'mixture_generator_samples_mode'])
            dataset_after_mutation = MixedGeneratorDataset(best_generators,
                                                           new_mixture_weights_generators,
                                                           self.score_sample_size,
                                                           self.cc.settings['trainer'][
                                                               'mixture_generator_samples_mode'])

            if self.score_calc is not None:
                self._logger.info('Calculating FID/inception score.')

                score_before_mutation = self.score_calc.calculate(dataset_before_mutation)[0]
                score_after_mutation = self.score_calc.calculate(dataset_after_mutation)[0]

                # For fid the lower the better, for inception_score, the higher the better
                if (score_after_mutation < score_before_mutation and self.score_calc.is_reversed) \
                        or (score_after_mutation > score_before_mutation and (not self.score_calc.is_reversed)):
                    # Adopt the mutated mixture_weights only if the performance after mutation is better
                    self.neighbourhood.mixture_weights_generators = new_mixture_weights_generators
                    self.score = score_after_mutation
                else:
                    # Do not adopt the mutated mixture_weights here
                    self.score = score_before_mutation

    def generate_random_fitness_samples(self, fitness_sample_size):
        """
        Generate random samples for fitness evaluation according to fitness_sample_size

        Abit of hack, use iterator of batch_size to sample data of fitness_sample_size
        TODO Implement another iterator (and dataloader) of fitness_sample_size
        """

        def get_next_batch(iterator, loaded):
            # Handle if the end of iterator is reached
            try:
                input, labels = next(iterator)
                return input, labels, iterator
            except StopIteration:
                # Use a new iterator
                iterator = iter(loaded)
                input, labels = next(iterator)
                return input, labels, iterator

        sampled_input, sampled_labels, self.fitness_iterator = get_next_batch(
            self.fitness_iterator, self.fitness_loaded)
        batch_size = sampled_input.size(0)

        if fitness_sample_size < batch_size:
            return sampled_input[:fitness_sample_size], sampled_labels[:fitness_sample_size]
        else:
            fitness_sample_size -= batch_size
            while fitness_sample_size >= batch_size:
                # Keep concatenate a full batch of data
                curr_input, curr_labels, self.fitness_iterator = get_next_batch(
                    self.fitness_iterator, self.fitness_loaded)
                sampled_input = torch.cat((sampled_input, curr_input), 0)
                sampled_labels = torch.cat((sampled_labels, curr_labels), 0)
                fitness_sample_size -= batch_size

            if fitness_sample_size > 0:
                # Concatenate partial batch of data
                curr_input, curr_labels, self.fitness_iterator = get_next_batch(
                    self.fitness_iterator, self.fitness_loaded)
                sampled_input = torch.cat(
                    (sampled_input, curr_input[:fitness_sample_size]), 0)
                sampled_labels = torch.cat(
                    (sampled_labels, curr_labels[:fitness_sample_size]), 0)

            return sampled_input, sampled_labels
