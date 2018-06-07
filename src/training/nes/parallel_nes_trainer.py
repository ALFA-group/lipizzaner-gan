import logging

import torch

from helpers.population import TYPE_GENERATOR, TYPE_DISCRIMINATOR
from helpers.pytorch_helpers import to_pytorch_variable
from training.nes.nes_trainer import NaturalEvolutionStrategyTrainer


class ParallelNESTrainer(NaturalEvolutionStrategyTrainer):

    _logger = logging.getLogger(__name__)

    def train(self, n_iterations, stop_event=None):
        loader = self.dataloader.load()

        self.lw_cache.init_session(n_iterations, self.population_gen, self.population_dis)

        for i in range(n_iterations):
            for j, (input_data, labels) in enumerate(loader):

                batch_size = input_data.size(0)
                input_var = to_pytorch_variable(input_data.view(batch_size, -1))

                self.evolve_generation(self.population_gen, self.population_dis, input_var)

                self.lw_cache.append_stepsizes(self.population_gen, self.population_dis)

                # If n_batches is set to 0, all batches will be used
                if self.dataloader.n_batches != 0 and self.dataloader.n_batches - 1 == j:
                    break

            self.lw_cache.log_best_individuals(i, self.population_gen, self.population_dis)
            self.log_results(batch_size, i, input_var, loader)

        self.lw_cache.end_session()

        return (self.population_gen.individuals[0].genome, self.population_gen.individuals[0].fitness), (
            self.population_dis.individuals[0].genome, self.population_dis.individuals[0].fitness)

    def evolve_generation(self, population_generator, population_discriminator, input_data):
            # Only one individual is evolved/kept per generation
            net_gen = population_generator.individuals[0].genome
            net_dis = population_discriminator.individuals[0].genome

            # Cache properties to avoid multiple recalculations
            params_gen = net_gen.parameters
            params_dis = net_dis.parameters

            # initialize memory for a population of w's, and their rewards
            # ziggurat implementation is used to enhance speed by factor ~10
            rands_gen = torch.randn(self._population_size, self._population_size, len(params_gen))
            rands_dis = torch.randn(self._population_size, self._population_size, len(params_dis))

            rewards_gen = torch.zeros(self._population_size, self._population_size)
            rewards_dis = torch.zeros(self._population_size, self._population_size)

            for i in range(self._population_size):
                for j in range(self._population_size):
                    w_try_gen = params_gen + self._sigma * rands_gen[i][j]
                    w_try_dis = params_dis + self._sigma * rands_dis[i][j]

                    rewards_gen[i][j] = self._evaluate_fitness(input_data, self.network_factory, TYPE_GENERATOR, w_try_gen,
                                                               w_try_dis)
                    rewards_dis[i][j] = self._evaluate_fitness(input_data, self.network_factory, TYPE_DISCRIMINATOR,
                                                               w_try_dis,
                                                               w_try_gen)

            # TODO: Enable different selection methods, choose via config file
            idx_gen, best_gen = self._best_population(rewards_gen)
            idx_dis, best_dis = self._best_population(rewards_dis)

            # standardize the rewards to have a gaussian distribution
            a_gen = (best_gen - best_gen.mean()) / best_gen.std()
            a_dis = (best_dis - best_dis.mean()) / best_dis.std()

            # perform the parameter update. The matrix multiply below
            # is just an efficient way to sum up all the rows of the noise matrix N,
            # where each row N[j] is weighted by A[j]
            net_gen.parameters = params_gen - self._alpha / (
                    self._population_size * self._sigma) * torch.matmul(rands_gen[idx_gen].t(), a_gen)
            net_dis.parameters = params_dis - self._alpha / (
                    self._population_size * self._sigma) * torch.matmul(rands_dis[idx_dis].t(), a_dis)

            population_generator.individuals[0].fitness = net_gen.compute_loss_against(net_dis, input_data)[0]
            population_discriminator.individuals[0].fitness = net_dis.compute_loss_against(net_gen, input_data)[0]

    @staticmethod
    def _best_population(rewards):
        _, idx = torch.min(torch.max(rewards, dim=1)[0], dim=0)
        return int(idx), rewards[int(idx)]

    @staticmethod
    def _evaluate_fitness(input_data, factory, net_type, w_try_self, w_try_opponent):

        self = factory.create_generator(
            w_try_self) if net_type == TYPE_GENERATOR else factory.create_discriminator(w_try_self)
        opponent = factory.create_generator(
            w_try_opponent) if net_type == TYPE_DISCRIMINATOR else factory.create_discriminator(w_try_opponent)
        return float(self.compute_loss_against(opponent, input_data)[0])
