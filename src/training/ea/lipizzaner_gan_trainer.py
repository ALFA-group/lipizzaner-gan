import random
from collections import OrderedDict
from time import time

import numpy as np
import torch
from data.network_data_loader import generate_random_sequences
from helpers.configuration_container import ConfigurationContainer
from helpers.db_logger import DbLogger
from helpers.population import TYPE_DISCRIMINATOR, TYPE_GENERATOR
from helpers.pytorch_helpers import noise, to_pytorch_variable
from training.ea.ea_trainer import EvolutionaryAlgorithmTrainer
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset
from training.mixture.score_factory import ScoreCalculatorFactory

from distribution.cell import Cell, Population

from copy import deepcopy


def evaluate_fitness(
    attackers: Population,
    defenders: Population,
    input_var,
    fitness_mode,
    labels=None,
    alpha=None,
    beta=None,
    iter=None,
    diverse_fitness=None,
):
    for attacker in attackers.values():
        attacker.fitness = 0
        for defender in defenders.values():
            (fitness,) = float(
                attacker.genome.compute_loss_against(
                    defender.genome,
                    input_var,
                    labels=labels,
                    alpha=alpha,
                    beta=beta,
                    iter=iter,
                    diverse_fitness=diverse_fitness,
                )
            )
            if fitness_mode == "best" and fitness > attacker.fitness:
                attacker.fitness = fitness
            elif fitness_mode == "worse" and fitness < attacker.fitness:
                attacker.fitness = fitness
            elif fitness_mode == "average":
                attacker.fitness = (fitness + attacker.fitness) / len(defenders)


class LipizzanerGANTrainer(EvolutionaryAlgorithmTrainer):
    """
    Distributed, asynchronous trainer for coevolutionary GANs. Uses the standard Goodfellow GAN approach.
    (Without discriminator mixture)
    """

    def __init__(
        self,
        dataloader,
        network_factory,
        cell: Cell,
        population_size=10,
        tournament_size=2,
        mutation_probability=0.9,
        n_replacements=1,
        sigma=0.25,
        alpha=0.25,
        default_adam_learning_rate=0.001,
        calc_mixture=False,
        mixture_sigma=0.01,
        score_sample_size=10000,
        discriminator_skip_each_nth_step=0,
        enable_selection=True,
        fitness_sample_size=10000,
        calculate_net_weights_dist=False,
        fitness_mode="worst",
        es_score_sample_size=10000,
        es_random_init=False,
        checkpoint_period=0,
    ):

        super().__init__(
            dataloader,
            network_factory,
            population_size,
            tournament_size,
            mutation_probability,
            n_replacements,
            sigma,
            alpha,
        )

        self.batch_number = 0
        self.cc = ConfigurationContainer.instance()

        self._default_adam_learning_rate = self.settings.get("default_adam_learning_rate", default_adam_learning_rate)
        self._discriminator_skip_each_nth_step = self.settings.get(
            "discriminator_skip_each_nth_step",
            discriminator_skip_each_nth_step,
        )
        self._enable_selection = self.settings.get("enable_selection", enable_selection)
        self.mixture_sigma = self.settings.get("mixture_sigma", mixture_sigma)

        self.cell = cell

        for i, individual in enumerate(self.population_gen.individuals):
            individual.learning_rate = self.settings.get(
                "default_g_adam_learning_rate",
                self._default_adam_learning_rate,
            )
            individual.id = "{}/G{}".format(self.neighbourhood.cell_number, i)
        for i, individual in enumerate(self.population_dis.individuals):
            individual.learning_rate = self.settings.get(
                "default_d_adam_learning_rate",
                self._default_adam_learning_rate,
            )
            individual.id = "{}/D{}".format(self.neighbourhood.cell_number, i)

        if "fitness" in self.settings:
            self.fitness_sample_size = self.settings["fitness"].get("fitness_sample_size", fitness_sample_size)
            self.fitness_loaded = self.dataloader.load()
            self.fitness_iterator = iter(self.fitness_loaded)  # Create iterator for fitness loader

            # Determine how to aggregate fitness calculated among neighbourhood
            self.fitness_mode = self.settings["fitness"].get("fitness_mode", fitness_mode)
            if self.fitness_mode not in ["worse", "best", "average"]:
                raise NotImplementedError("Invalid argument for fitness_mode: {}".format(self.fitness_mode))
        else:
            # TODO: Add code for safe implementation & error handling
            raise KeyError("Fitness section must be defined in configuration file")

        n_iterations = self.cc.settings["trainer"].get("n_iterations", 0)

        if (
            "score" in self.settings and self.settings["score"].get("enabled", calc_mixture)
        ) or "optimize_mixture" in self.settings:
            self.score_calc = ScoreCalculatorFactory.create()
            self.score_sample_size = self.settings["score"].get("sample_size", score_sample_size)
            self.score = float("inf") if self.score_calc.is_reversed else float("-inf")
            self.mixture_generator_samples_mode = self.cc.settings["trainer"]["mixture_generator_samples_mode"]
        else:
            self.score_sample_size = score_sample_size
            self.score_calc = None
            self.score = 0

        if "optimize_mixture" in self.settings:
            self.optimize_weights_at_the_end = self.settings["optimize_mixture"].get("enabled", True)
            self.score_sample_size = self.settings["optimize_mixture"].get("sample_size", es_score_sample_size)
            self.es_generations = self.settings["optimize_mixture"].get("es_generations", n_iterations)
            self.es_random_init = self.settings["optimize_mixture"].get("es_random_init", es_random_init)
            self.mixture_sigma = self.settings["optimize_mixture"].get("mixture_sigma", mixture_sigma)
            self.mixture_generator_samples_mode = self.cc.settings["trainer"]["mixture_generator_samples_mode"]
        else:
            self.optimize_weights_at_the_end = True
            self.score_sample_size = es_score_sample_size
            self.es_generations = n_iterations
            self.es_random_init = es_random_init
            self.mixture_sigma = mixture_sigma
            self.mixture_generator_samples_mode = self.cc.settings["trainer"]["mixture_generator_samples_mode"]

        assert 0 <= checkpoint_period <= n_iterations, (
            "Checkpoint period paramenter (checkpoint_period) should be "
            "between 0 and the number of iterations (n_iterations)."
        )
        self.checkpoint_period = self.cc.settings["general"].get("checkpoint_period", checkpoint_period)

        self.alpha = self.neighbourhood.alpha
        self.beta = self.neighbourhood.beta

        self.diverse_fitness = self.cc.settings["trainer"]["params"]["fitness"].get("diverse_fitness", None)

    def _run_generation(
        self,
        iteration,
    ):
        self._logger.debug("Iteration {} started".format(iteration + 1))
        start_time = time()

        if self.neighbourhood.scheduler is not None and iteration in self.neighbourhood.scheduler:
            alpha = self.neighbourhood.scheduler[str(iteration)]["alpha"]
            beta = self.neighbourhood.scheduler[str(iteration)]["beta"]

        all_generators = self.cell.generators
        all_discriminators = self.cell.discriminators
        local_generators = self.cell.generators.center
        local_discriminators = self.cell.discriminators.center

        # Log the name of individuals in entire neighborhood and local individuals for every iteration
        # (to help tracing because individuals from adjacent cells might be from different iterations)
        self._logger.info("Neighborhood located in possition {} of the grid".format(self.neighbourhood.grid_position))
        self._logger.info(
            "Generators in current neighborhood are {}".format(
                [individual.name for individual in all_generators.individuals]
            )
        )
        self._logger.info(
            "Discriminators in current neighborhood are {}".format(
                [individual.name for individual in all_discriminators.individuals]
            )
        )
        self._logger.info(
            "Local generators in current neighborhood are {}".format(
                [individual.name for individual in local_generators.individuals]
            )
        )
        self._logger.info(
            "Local discriminators in current neighborhood are {}".format(
                [individual.name for individual in local_discriminators.individuals]
            )
        )

        self._logger.info("L2 distance between all generators weights: {}".format(all_generators.net_weights_dist))
        self._logger.info(
            "L2 distance between all discriminators weights: {}".format(all_discriminators.net_weights_dist)
        )

        new_populations = {}

        # Create random dataset to evaluate fitness in each iterations
        (
            fitness_samples,
            fitness_labels,
        ) = self.generate_random_fitness_samples(self.fitness_sample_size)
        if (
            self.cc.settings["dataloader"]["dataset_name"] == "celeba"
            or self.cc.settings["dataloader"]["dataset_name"] == "cifar"
            or self.cc.settings["network"]["name"] == "ssgan_convolutional_mnist"
        ):
            fitness_samples = to_pytorch_variable(fitness_samples)
            fitness_labels = to_pytorch_variable(fitness_labels)
        elif self.cc.settings["dataloader"]["dataset_name"] == "network_traffic":
            fitness_samples = to_pytorch_variable(generate_random_sequences(self.fitness_sample_size))
        else:
            fitness_samples = to_pytorch_variable(fitness_samples.view(self.fitness_sample_size, -1))
            fitness_labels = to_pytorch_variable(fitness_labels.view(self.fitness_sample_size, -1))

        fitness_labels = torch.squeeze(fitness_labels)

        # Fitness evaluation
        self._logger.debug("Evaluating fitness")
        evaluate_fitness(
            all_generators,
            all_discriminators,
            fitness_samples,
            self.fitness_mode,
            alpha=alpha,
            beta=beta,
            diverse_fitness=self.diverse_fitness,
        )
        evaluate_fitness(
            all_discriminators,
            all_generators,
            fitness_samples,
            self.fitness_mode,
            labels=fitness_labels,
            logger=self._logger,
            iter=iteration,
            log_class_distribution=True,
        )
        self._logger.debug("Finished evaluating fitness")

        # Tournament selection
        if self._enable_selection:
            self._logger.debug("Started tournament selection")
            new_populations[TYPE_GENERATOR] = self.tournament_selection(all_generators, TYPE_GENERATOR, is_logging=True)
            new_populations[TYPE_DISCRIMINATOR] = self.tournament_selection(
                all_discriminators, TYPE_DISCRIMINATOR, is_logging=True
            )
            self._logger.debug("Finished tournament selection")

        self.batch_number = 0
        data_iterator = iter(self.loaded)
        while self.batch_number < len(self.loaded):
            if self.cc.settings["dataloader"]["dataset_name"] == "network_traffic":
                input_data = to_pytorch_variable(next(data_iterator))
                batch_size = input_data.size(0)
            else:
                input_data, labels = next(data_iterator)
                batch_size = input_data.size(0)
                input_data = to_pytorch_variable(self.dataloader.transpose_data(input_data))
                labels = to_pytorch_variable(self.dataloader.transpose_data(labels))
                labels = torch.squeeze(labels)

            attackers = new_populations[TYPE_GENERATOR] if self._enable_selection else local_generators
            defenders = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else all_discriminators

            attackers = [deepcopy(network) for network in attackers]
            defenders = [deepcopy(network) for network in defenders]

            input_data = self.step(
                local_generators,
                attackers,
                defenders,
                input_data,
                self.batch_number,
                self.loaded,
                data_iterator,
                iter=iteration,
            )

            if (
                self._discriminator_skip_each_nth_step == 0
                or self.batch_number % (self._discriminator_skip_each_nth_step + 1) == 0
            ):
                self._logger.debug("Skipping discriminator step")

                attackers = new_populations[TYPE_DISCRIMINATOR] if self._enable_selection else local_discriminators
                defenders = new_populations[TYPE_GENERATOR] if self._enable_selection else all_generators
                input_data = self.step(
                    local_discriminators,
                    attackers,
                    defenders,
                    input_data,
                    self.batch_number,
                    self.loaded,
                    data_iterator,
                    labels=labels,
                    alpha=alpha,
                    beta=beta,
                    iter=iteration,
                )

            self._logger.info("Iteration {}, Batch {}/{}".format(iteration + 1, self.batch_number, len(self.loaded)))

            # If n_batches is set to 0, all batches will be used
            if self.is_last_batch(self.batch_number):
                break

            self.batch_number += 1

        # Perform selection first before mutation of mixture_weights
        # Replace the worst with the best new
        if self._enable_selection:
            # Evaluate fitness of new_populations against neighborhood
            evaluate_fitness(
                new_populations[TYPE_GENERATOR],
                all_discriminators,
                fitness_samples,
                self.fitness_mode,
                alpha=alpha,
                beta=beta,
                diverse_fitness=self.diverse_fitness,
            )
            evaluate_fitness(
                new_populations[TYPE_DISCRIMINATOR],
                all_generators,
                fitness_samples,
                self.fitness_mode,
                labels=fitness_labels,
                iter=iteration,
            )
            self.concurrent_populations.lock()
            local_generators.replacement(
                new_populations[TYPE_GENERATOR],
                self._n_replacements,
                is_logging=True,
            )
            local_generators.sort_population(is_logging=True)
            local_discriminators.replacement(
                new_populations[TYPE_DISCRIMINATOR],
                self._n_replacements,
                is_logging=True,
            )
            local_discriminators.sort_population(is_logging=True)
            self.concurrent_populations.unlock()

            # Update individuals' iteration and id after replacement and logging to ease tracing
            for i, individual in enumerate(local_generators.individuals):
                individual.id = "{}/G{}".format(self.neighbourhood.cell_number, i)
                individual.iteration = iteration + 1
            for i, individual in enumerate(local_discriminators.individuals):
                individual.id = "{}/D{}".format(self.neighbourhood.cell_number, i)
                individual.iteration = iteration + 1
        else:
            # Re-evaluate fitness of local_generators and local_discriminators against neighborhood
            evaluate_fitness(
                local_generators,
                all_discriminators,
                fitness_samples,
                self.fitness_mode,
                alpha=alpha,
                beta=beta,
                diverse_fitness=self.diverse_fitness,
            )
            evaluate_fitness(
                local_discriminators,
                all_generators,
                fitness_samples,
                self.fitness_mode,
                labels=fitness_labels,
                iter=iteration,
            )

        self.compute_mixture_generative_score(iteration)

        stop_time = time()

        path_real_images, path_fake_images = self.log_results(
            batch_size,
            iteration,
            input_data,
            self.loaded,
            lr_gen=self.concurrent_populations.generator.individuals[0].learning_rate,
            lr_dis=self.concurrent_populations.discriminator.individuals[0].learning_rate,
            score=self.score,
            mixture_gen=self.neighbourhood.mixture_weights_generators,
            mixture_dis=None,
        )

        if self.db_logger.is_enabled:
            self.db_logger.log_results(
                iteration,
                self.neighbourhood,
                self.concurrent_populations,
                self.score,
                stop_time - start_time,
                path_real_images,
                path_fake_images,
            )

        if self.checkpoint_period > 0 and (iteration + 1) % self.checkpoint_period == 0:
            self.save_checkpoint(
                all_generators.individuals,
                all_discriminators.individuals,
                self.neighbourhood.cell_number,
                self.neighbourhood.grid_position,
            )

    def _prepare_data(self):
        self.loaded = self.dataloader.load()

    def _optimize_generator_mixture(self, n_iterations, batch_size, input_data):
        if self.optimize_weights_at_the_end:

            start_time = time()
            self.optimize_generator_mixture_weights()
            stop_time = time()

            path_real_images, path_fake_images = self.log_results(
                batch_size,
                n_iterations,
                input_data,
                self.loaded,
                lr_gen=self.concurrent_populations.generator.individuals[0].learning_rate,
                lr_dis=self.concurrent_populations.discriminator.individuals[0].learning_rate,
                score=self.score,
                mixture_gen=self.neighbourhood.mixture_weights_generators,
                mixture_dis=self.neighbourhood.mixture_weights_discriminators,
            )

            if self.db_logger.is_enabled:
                self.db_logger.log_results(
                    n_iterations,
                    self.neighbourhood,
                    self.concurrent_populations,
                    self.score,
                    stop_time - start_time,
                    path_real_images,
                    path_fake_images,
                )

    def test_majority_voting_discriminators(self, models, test_loader, train=False):
        correct = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_or_test = "Train" if train else "Test"
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if self.cc.settings["network"]["name"] == "ssgan_perceptron":
                    data = data.view(-1, 784)
                elif self.cc.settings["network"]["name"] == "ssgan_conv_mnist_28x28":
                    data = data.view(-1, 1, 28, 28)
                elif self.cc.settings["network"]["name"] == "ssgan_svhn":
                    data = data.view(-1, 3, 32, 32)
                else:
                    if self.cc.settings["dataloader"]["dataset_name"] == "cifar":
                        data = data.view(-1, 3, 64, 64)
                    else:
                        data = data.view(-1, 1, 64, 64)
                pred_accumulator = []
                for model in models:
                    output = model.classification_layer(model.net(data))
                    output = output.view(-1, 11)
                    pred = output.argmax(dim=1, keepdim=True)
                    pred_accumulator.append(pred.view(-1))
                label_votes = to_pytorch_variable(torch.tensor(list(zip(*pred_accumulator))))
                prediction = to_pytorch_variable(
                    torch.tensor([labels.bincount(minlength=11).argmax() for labels in label_votes])
                )
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        num_samples = len(test_loader.dataset)
        accuracy = 100.0 * float(correct / num_samples)
        self._logger.info(f"Majority Voting {train_or_test} Accuracy: {correct}/{num_samples} ({accuracy}%)")

    def test_accuracy_discriminators(self, model, test_loader, train=False):
        correct = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_or_test = "Train" if train else "Test"
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if self.cc.settings["network"]["name"] == "ssgan_perceptron":
                    data = data.view(-1, 784)
                elif self.cc.settings["network"]["name"] == "ssgan_conv_mnist_28x28":
                    data = data.view(-1, 1, 28, 28)
                elif self.cc.settings["network"]["name"] == "ssgan_svhn":
                    data = data.view(-1, 3, 32, 32)
                elif self.cc.settings["dataloader"]["dataset_name"] == "cifar":
                    data = data.view(-1, 3, 64, 64)
                else:
                    data = data.view(-1, 1, 64, 64)
                output = model.classification_layer(model.net(data))
                output = output.view(-1, 11)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        num_samples = len(test_loader.dataset)
        accuracy = 100.0 * float(correct / num_samples)
        self._logger.info(f"{train_or_test} Accuracy: {correct}/{num_samples} ({accuracy}%)")

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

        dataset = MixedGeneratorDataset(
            generators,
            weights_generators,
            self.score_sample_size,
            self.mixture_generator_samples_mode,
            z_noise,
        )

        self.score = self.score_calc.calculate(dataset)[0]
        init_score = self.score

        self._logger.info("Mixture weight mutation - Starting mixture weights optimization ...")
        self._logger.info("Init score: {}\tInit weights: {}.".format(init_score, weights_generators))

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
            dataset = MixedGeneratorDataset(
                generators,
                new_mixture_weights,
                self.score_sample_size,
                self.mixture_generator_samples_mode,
                z_noise,
            )

            if self.score_calc is not None:
                score_after_mutation = self.score_calc.calculate(dataset)[0]
                self._logger.info(
                    "Mixture weight mutation - Generation: {} \tScore of new weights: {}\tNew weights: {}.".format(
                        g, score_after_mutation, new_mixture_weights
                    )
                )

                # For fid the lower the better, for inception_score, the higher the better
                if (score_after_mutation < self.score and self.score_calc.is_reversed) or (
                    score_after_mutation > self.score and (not self.score_calc.is_reversed)
                ):
                    weights_generators = new_mixture_weights
                    self.score = score_after_mutation
                    self._logger.info(
                        "Mixture weight mutation - Generation: {} \tNew score: {}\tWeights changed to: {}.".format(
                            g, self.score, weights_generators
                        )
                    )
        self.neighbourhood.mixture_weights_generators = weights_generators

        self._logger.info(
            "Mixture weight mutation - Score before mixture weight optimzation: {}\tScore after mixture weight optimzation: {}.".format(
                init_score, self.score
            )
        )

    def step(
        self,
        original,
        attacker,
        defender,
        input_data,
        i,
        data_iterator,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
    ):
        self.mutate_hyperparams(attacker)
        return self.update_genomes(
            attacker,
            defender,
            input_data,
            self.loaded,
            data_iterator,
            labels=labels,
            alpha=alpha,
            beta=beta,
            iter=iter,
        )

    def is_last_batch(self, i):
        return self.dataloader.n_batches != 0 and self.dataloader.n_batches - 1 == i

    def result(self):
        return (
            (
                self.concurrent_populations.generator.individuals[0].genome,
                self.concurrent_populations.generator.individuals[0].fitness,
            ),
            (
                self.concurrent_populations.discriminator.individuals[0].genome,
                self.concurrent_populations.discriminator.individuals[0].fitness,
            ),
        )

    def mutate_hyperparams(self, population):
        loc = -(self._default_adam_learning_rate / 10)
        deltas = np.random.normal(
            loc=loc,
            scale=self._default_adam_learning_rate,
            size=len(population.individuals),
        )
        deltas[np.random.rand(*deltas.shape) < 1 - self._mutation_probability] = 0
        for i, individual in enumerate(population.individuals):
            individual.learning_rate = max(0, individual.learning_rate + deltas[i] * self._alpha)

    def update_genomes(
        self,
        population_attacker,
        population_defender,
        input_var,
        data_iterator,
        labels=None,
        alpha=None,
        beta=None,
        iter=None,
    ):

        # TODO Currently picking random opponent, introduce parameter for this
        defender = random.choice(population_defender.individuals).genome

        for individual_attacker in population_attacker.individuals:
            attacker = individual_attacker.genome
            optimizer = torch.optim.Adam(
                attacker.net.parameters(),
                lr=individual_attacker.learning_rate,
                betas=(0.5, 0.999),
            )

            # Restore previous state dict, if available
            if individual_attacker.optimizer_state is not None:
                optimizer.load_state_dict(individual_attacker.optimizer_state)

            if labels is None:
                loss = attacker.compute_loss_against(defender, input_var)[0]
            else:
                loss = attacker.compute_loss_against(
                    defender,
                    input_var,
                    labels=labels,
                    alpha=alpha,
                    beta=beta,
                    iter=iter,
                )[0]

            attacker.net.zero_grad()
            defender.net.zero_grad()
            loss.backward()
            optimizer.step()

            individual_attacker.optimizer_state = optimizer.state_dict()

        return input_var

    def compute_mixture_generative_score(self, iteration):
        # Not necessary for single-cell grids, as mixture must always be [1]
        self._logger.info("Calculating score.")
        best_generators = self.neighbourhood.best_generators

        if True or self.neighbourhood.grid_size == 1:
            if self.score_calc is not None:
                dataset = MixedGeneratorDataset(
                    best_generators,
                    self.neighbourhood.mixture_weights_generators,
                    self.score_sample_size,
                    self.cc.settings["trainer"]["mixture_generator_samples_mode"],
                )
                self.score = self.score_calc.calculate(dataset)
                self._logger.info(f"Score for iteration {iteration}: {self.score} ")

    def generate_random_fitness_samples(self, fitness_sample_size):
        """
        Generate random samples for fitness evaluation according to fitness_sample_size

        Abit of hack, use iterator of batch_size to sample data of fitness_sample_size
        TODO Implement another iterator (and dataloader) of fitness_sample_size
        """

        def get_next_batch(iterator):
            iterator = self.fitness_iterator
            loaded = self.fitness_loaded
            # Handle if the end of iterator is reached
            try:
                input, labels = next(iterator)
                return input, labels, iterator
            except StopIteration:
                # Use a new iterator
                iterator = iter(loaded)
                input, labels = next(iterator)
                return input, labels, iterator

        sampled_data, sampled_labels, self.fitness_iterator = get_next_batch()
        batch_size = sampled_data.size(0)

        if fitness_sample_size < batch_size:
            return (
                sampled_data[:fitness_sample_size],
                sampled_labels[:fitness_sample_size],
            )
        else:
            fitness_sample_size -= batch_size
            while fitness_sample_size >= batch_size:
                # Keep concatenate a full batch of data
                curr_data, curr_labels, self.fitness_iterator = get_next_batch()
                sampled_data = torch.cat((sampled_data, curr_data), 0)
                sampled_labels = torch.cat((sampled_labels, curr_labels), 0)
                fitness_sample_size -= batch_size

            if fitness_sample_size > 0:
                # Concatenate partial batch of data
                curr_data, curr_labels, self.fitness_iterator = get_next_batch()
                sampled_data = torch.cat((sampled_data, curr_data[:fitness_sample_size]), 0)
                sampled_labels = torch.cat((sampled_labels, curr_labels[:fitness_sample_size]), 0)

            return sampled_data, sampled_labels
