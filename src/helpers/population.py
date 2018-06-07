TYPE_GENERATOR = 0
TYPE_DISCRIMINATOR = 1

ALWAYS_REPLACE = True


class Population:

    def __init__(self, individuals, default_fitness, population_type=None):
        self.individuals = individuals
        self.default_fitness = default_fitness
        self.population_type = population_type

    def sort_population(self):
        self.individuals.sort(key=lambda x: x.fitness)

    def replacement(self, new_population, n_replacements=1):
        new_population.sort_population()
        self.sort_population()
        # TODO break out early
        for i in range(n_replacements):
            j = i - n_replacements
            if ALWAYS_REPLACE or self.individuals[j].fitness > new_population.individuals[i].fitness:
                self.individuals[j] = new_population.individuals[i].clone()
                self.individuals[j].is_local = True
