from helpers.pytorch_helpers import to_pytorch_variable
from training.ea.ea_trainer import EvolutionaryAlgorithmTrainer


class AlternatingEATrainer(EvolutionaryAlgorithmTrainer):
    def train(self, n_iterations, stop_event=None):

        loader = self.dataloader.load()

        self.lw_cache.init_session(n_iterations, self.population_gen, self.population_dis)

        for i in range(n_iterations):
            for j, (input_data, labels) in enumerate(loader):

                batch_size = input_data.size(0)
                input_var = to_pytorch_variable(input_data.view(batch_size, -1))

                if i == 0:
                    self.evaluate_fitness_against_population(self.population_gen, self.population_dis, input_var)

                for attacker, defender in ((self.population_gen, self.population_dis),
                                           (self.population_dis, self.population_gen)):
                    new_population = self.tournament_selection(attacker)
                    self.mutate_gaussian(new_population)
                    self.evaluate_fitness_against_population(new_population, defender, input_var)

                    # Replace the worst with the best new
                    attacker.replacement(new_population, self._n_replacements)
                    attacker.sort_population()

                self.lw_cache.append_stepsizes(self.population_gen, self.population_dis)

                # If n_batches is set to 0, all batches will be used
                if self.dataloader.n_batches != 0 and self.dataloader.n_batches - 1 == j:
                    break

            self.lw_cache.log_best_individuals(i, self.population_gen, self.population_dis)
            self.log_results(batch_size, i, input_var, loader)

        self.lw_cache.end_session()

        return (self.population_gen.individuals[0].genome, self.population_gen.individuals[0].fitness), (
            self.population_dis.individuals[0].genome, self.population_dis.individuals[0].fitness)
