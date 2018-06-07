import logging

import numpy as np
import torch

from helpers.population import TYPE_GENERATOR
from helpers.pytorch_helpers import to_pytorch_variable, save_images
from training.nes.nes_trainer import NaturalEvolutionStrategyTrainer


class SequentialNESTrainer(NaturalEvolutionStrategyTrainer):

    _logger = logging.getLogger(__name__)

    def train(self, n_iterations, stop_event=None):
        loader = self.dataloader.load()

        distributor = Distributor()

        if self._population_size % distributor.n_procs_overall != 0:
            self._logger.error("Number of overall processes should be a factor of population size, "
                               "distribution may not work.")

        self.lw_cache.init_session(n_iterations, self.population_gen, self.population_dis)

        for i in range(n_iterations):
            for j, (input_data, labels) in enumerate(loader):

                batch_size = input_data.size(0)
                input_var = to_pytorch_variable(input_data.view(batch_size, -1))

                populations = self.population_gen, self.population_dis
                self.evolve_generation(distributor, populations, input_var)

                self.lw_cache.append_stepsizes(self.population_gen, self.population_dis)

                # If n_batches is set to 0, all batches will be used
                if self.dataloader.n_batches != 0 and self.dataloader.n_batches - 1 == j:
                    break

            self.lw_cache.log_best_individuals(i, self.population_gen, self.population_dis)
            self.log_results(batch_size, i, input_var, loader)

        self.lw_cache.end_session()

        return (self.population_gen.individuals[0].genome, self.population_gen.individuals[0].fitness), (
            self.population_dis.individuals[0].genome, self.population_dis.individuals[0].fitness)

    def evolve_generation(self, distributor, populations, input_data):
        for net_type, population in enumerate(populations):

            params = population.individuals[0].genome.parameters
            opponent = populations[1 - net_type].individuals[0]

            index_chunks = np.array_split(range(self._population_size), distributor.n_procs_overall)
            output = distributor.run_distributed(self._calc_rewards, index_chunks=index_chunks,
                                                 params=params, opponent=opponent, input_data=input_data,
                                                 net_type=net_type)

            if distributor.is_master_node:
                rewards, rands = output

                # standardize the rewards to have a gaussian distribution
                a = (rewards - rewards.mean()) / rewards.std()

                # perform the parameter update. The matrix multiply below
                # is just an efficient way to sum up all the rows of the noise matrix N,
                # where each row N[j] is weighted by A[j]
                population.individuals[0].genome.parameters = params - self._alpha / (
                        self._population_size * self._sigma) * torch.matmul(rands.t(), a)
                population.individuals[0].fitness = \
                    population.individuals[0].genome.compute_loss_against(opponent.genome, input_data)[0]

    def _calc_rewards(self, distributor, worker_index, set_output, index_chunks, params, opponent, input_data, net_type):
        clone = self._create_clone(self.network_factory, net_type)

        chunk_len = len(index_chunks[worker_index])
        rands = torch.randn(chunk_len, len(params))
        rewards = torch.zeros(chunk_len)

        for i in range(chunk_len):
            w_try = params + self._sigma * rands[i]
            rewards[i] = self._evaluate_fitness(input_data, opponent, clone, w_try)

        list_rewards = [torch.zeros(chunk_len)] * distributor.n_procs_overall
        list_randoms = [torch.zeros(chunk_len, len(params))] * distributor.n_procs_overall
        distributor.all_gather(rewards, list_rewards)
        distributor.all_gather(rands, list_randoms)

        # Only master node has to set the output
        if worker_index == 0:
            set_output((torch.cat(list_rewards), torch.cat(list_randoms)))

    @staticmethod
    def _create_clone(network_factory, net_type):
        return network_factory.create_generator() if net_type == TYPE_GENERATOR else \
            network_factory.create_discriminator()

    @staticmethod
    def _evaluate_fitness(input_data, opponent, clone, w_try):
        clone.parameters = w_try
        return float(clone.compute_loss_against(opponent.genome, input_data)[0])

    @staticmethod
    def _round_down(num, divisor):
        return num - (num % divisor)