from helpers.pytorch_helpers import to_pytorch_variable
from helpers.population import TYPE_GENERATOR, TYPE_DISCRIMINATOR
from training.ea.ea_trainer import EvolutionaryAlgorithmTrainer


class ParallelEATrainer(EvolutionaryAlgorithmTrainer):
    def train(self, n_iterations, stop_event=None):

        loader = self.dataloader.load()

        self.lw_cache.init_session(n_iterations, self.population_gen, self.population_dis)

        for i in range(n_iterations):
            for j, (input_data, labels) in enumerate(loader):

                batch_size = input_data.size(0)
                input_var = to_pytorch_variable(input_data.view(batch_size, -1))

                if i == 0 and j == 0:
                    self.evaluate_fitness_against_population(self.population_gen, self.population_dis, input_var)

                new_population_generator = self.tournament_selection(self.population_gen, TYPE_GENERATOR)
                new_population_discriminator = self.tournament_selection(self.population_dis, TYPE_DISCRIMINATOR)
                self.mutate_gaussian(new_population_generator)
                self.mutate_gaussian(new_population_discriminator)

                self.evaluate_fitness_against_population(new_population_generator, new_population_discriminator, input_var)

                self.population_gen.replacement(new_population_generator, self._n_replacements)
                self.population_dis.replacement(new_population_discriminator, self._n_replacements)
                self.population_gen.sort_population()
                self.population_dis.sort_population()

                self.lw_cache.append_stepsizes(self.population_gen, self.population_dis)

                if self.dataloader.n_batches != 0 and self.dataloader.n_batches - 1 == j:
                    break

            self.lw_cache.log_best_individuals(i, self.population_gen, self.population_dis)
            self.log_results(batch_size, i, input_var, loader)

        self.lw_cache.end_session()

        return (self.population_gen.individuals[0].genome, self.population_gen.individuals[0].fitness), (
            self.population_dis.individuals[0].genome, self.population_dis.individuals[0].fitness)
