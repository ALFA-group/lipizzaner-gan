import concurrent
import csv
import math
import os.path
import random
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import time
from scipy.stats import norm

from opt_disc import find_optimal_discriminator

NEGATIVE = -1
POSITIVE = 1

DISCRIMINATOR_INIT_DISTANCE = 0.2

DEFAULT_FITNESS_MINIMIZER = float('inf')
DEFAULT_FITNESS_MAXIMIZER = float('-inf')
DISCRIMINATOR = 'discriminator'
GENERATOR = 'generator'
DEFAULT_FITNESSES = {GENERATOR: DEFAULT_FITNESS_MINIMIZER,
                     DISCRIMINATOR: DEFAULT_FITNESS_MAXIMIZER,
                     }
POPULATION_NAMES = (DISCRIMINATOR, GENERATOR)
SORT_ORDERS = {GENERATOR: False,
               DISCRIMINATOR: True}


class Individual(object):
    """
    Genome is a vector of the following elements
    - Generator: [µ1, delta µ]
      where µ1 + delta µ = µ2

    - Discriminator: [l1, delta1, delta2, delta3]
      where l1 + delta1 = r1,
      where r1 + delta2 = l2,
      where l2 + delta3 = r2,
    """

    def __init__(self, fitness, genome, population_size):
        self.genome = genome
        self.fitness = fitness
        self.adversary_solution = None
        self.all_fitnesses = [None] * population_size

    def __str__(self):
        return "fitness={} genome={} adv={}".format(self.fitness, self.genome, self.adversary_solution)


class Population(object):

    def __init__(self, sort_order, individuals, name, default_fitness):
        self.sort_order = sort_order
        self.individuals = individuals
        self.name = name
        self.default_fitness = default_fitness

        if self.sort_order:
            self.compare = lambda x, y: x < y
        else:
            self.compare = lambda x, y: x > y

    def sort_population(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=self.sort_order)

    def replacement(self, new_population, n_replacements=1):
        new_population.sort_population()
        self.sort_population()
        # TODO break out early
        for i in range(n_replacements):
            j = i - n_replacements
            if self.compare(self.individuals[j].fitness, new_population.individuals[i].fitness):
                self.individuals[j] = new_population.individuals[i]

    def __str__(self):
        return "{} {}".format(self.name, ', '.join(map(str, self.individuals)))


class MinMaxMethod(object):
    """
        Python interface for minmax methods
    """

    def __init__(self, fct, max_fevals, seed):
        self._fct = fct
        self._max_fevals = max_fevals
        self._seed = seed
        if seed != 0:
            np.random.seed(self._seed)
            random.seed(self._seed)


class Coev(MinMaxMethod):

    def __init__(self, fct, m1_opt, m2_opt, max_bounds, gaussian_step, mutation_probability=0.9, population_size=10,
                 tournament_size=2, n_replacements=1, max_fevals=100, seed=1, verbose=False, evaluate_asymmetric=False,
                 fix_discriminator=False, fix_generator=False, optimal_discriminator=False, m1_init=None, m2_init=None,
                 discriminator_collapse=False, experiment_number=None, detailed_log=True,
                 discriminator_collapse_type=None):
        super(Coev, self).__init__(fct, max_fevals=max_fevals, seed=seed)
        self.discriminator_collapse_type = discriminator_collapse_type
        self.detailed_log = detailed_log
        self.experiment_number = experiment_number
        self.m1_init = m1_init
        self.m2_init = m2_init
        self.discriminator_collapse = discriminator_collapse
        self.fix_discriminator = fix_discriminator
        self.fix_generator = fix_generator
        self.optimal_discriminator = optimal_discriminator
        self.evaluate_asymmetric = evaluate_asymmetric
        self.gaussian_step = gaussian_step
        self.max_bounds = max_bounds
        self.m1_opt = m1_opt
        self.m2_opt = m2_opt
        self.verbose = verbose
        self.T = max_fevals // (population_size ** 2)
        print("Running for {} iterations".format(self.T))

        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.n_replacements = n_replacements

        self.results = []

    def run(self):
        raise NotImplementedError

    def log_results(self, populations, generation):
        if self.detailed_log:
            self.log_detailed(populations, generation)
        else:
            self.log_undetailed(populations, generation)

    def log_detailed(self, populations, generation):
        for i, individual_gen in enumerate(populations[GENERATOR].individuals):
            for j, individual_dis in enumerate(populations[DISCRIMINATOR].individuals):
                self.results.append(defaultdict(list))

                ms = np.cumsum(individual_gen.genome)
                self.results[-1]['round'] = self.experiment_number
                self.results[-1]['generation'] = generation
                self.results[-1]['individual_gen'] = i
                self.results[-1]['individual_dis'] = j

                self.results[-1]['m1_opt'] = self.m1_opt
                self.results[-1]['m2_opt'] = self.m2_opt
                self.results[-1]['m1'] = ms[0]
                self.results[-1]['m2'] = ms[1]

                bounds = np.cumsum(individual_gen.adversary_solution)
                self.results[-1]['gen_adversarial_l1'] = bounds[0]
                self.results[-1]['gen_adversarial_r1'] = bounds[1]
                self.results[-1]['gen_adversarial_l2'] = bounds[2]
                self.results[-1]['gen_adversarial_r2'] = bounds[3]

                bounds = np.cumsum(individual_dis.genome)
                self.results[-1]['current_discr_l1'] = bounds[0]
                self.results[-1]['current_discr_r1'] = bounds[1]
                self.results[-1]['current_discr_l2'] = bounds[2]
                self.results[-1]['current_discr_r2'] = bounds[3]

                self.results[-1]['objective_best_fitness'] = individual_gen.fitness
                self.results[-1]['subjective_fitness_gen'] = individual_gen.all_fitnesses[i]
                self.results[-1]['subjective_fitness_dis'] = individual_dis.all_fitnesses[j]

                if self.m1_init is not None and self.m2_init is not None:
                    self.results[-1]['m1_init'] = self.m1_init
                    self.results[-1]['m2_init'] = self.m2_init

    def log_undetailed(self, populations, generation):
        for i, individual_gen in enumerate(populations[GENERATOR].individuals):
            self.results.append(defaultdict(list))

            ms = np.cumsum(individual_gen.genome)
            self.results[-1]['round'] = self.experiment_number
            self.results[-1]['generation'] = generation
            self.results[-1]['individual_gen'] = i

            self.results[-1]['m1_opt'] = self.m1_opt
            self.results[-1]['m2_opt'] = self.m2_opt
            self.results[-1]['m1'] = ms[0]
            self.results[-1]['m2'] = ms[1]

            bounds = np.cumsum(individual_gen.adversary_solution)
            self.results[-1]['gen_adversarial_l1'] = bounds[0]
            self.results[-1]['gen_adversarial_r1'] = bounds[1]
            self.results[-1]['gen_adversarial_l2'] = bounds[2]
            self.results[-1]['gen_adversarial_r2'] = bounds[3]

            self.results[-1]['objective_best_fitness'] = individual_gen.fitness
            self.results[-1]['subjective_fitness_gen'] = individual_gen.all_fitnesses[i]

            if self.m1_init is not None and self.m2_init is not None:
                self.results[-1]['m1_init'] = self.m1_init
                self.results[-1]['m2_init'] = self.m2_init

            if self.discriminator_collapse_type is not None:
                self.results[-1]['discr_collapse_type'] = self.discriminator_collapse_type

    def initialize_populations(self):
        populations = self._populate()

        if self.fix_discriminator:
            for ind in populations[DISCRIMINATOR].individuals:
                ind.genome[0] = -3
                ind.genome[1] = 2
                ind.genome[2] = 2
                ind.genome[3] = 2

        if self.fix_generator:
            for ind in populations[GENERATOR].individuals:
                ind.genome[0] = -4
                ind.genome[1] = 4

        # Set bounds for discriminator collapse
        if self.discriminator_collapse:
            done = False
            while not done:
                try:
                    self.create_discr_collapse(populations)
                    done = True
                except:
                    populations = self._populate()

        return populations

    def create_discr_collapse(self, populations):
        negative_bounds = []
        # print("Initializing populations for discr_collapse")
        # Get optimal discriminator for all generators
        optimal_disc = []
        for ind in populations[GENERATOR].individuals:
            ms = np.cumsum(ind.genome)
            params = {'p_mu_1': self.m1_opt, 'p_mu_2': self.m2_opt, 'q_mu_1': ms[0], 'q_mu_2': ms[1]}
            optimal_disc.append(find_optimal_discriminator(params))
        l1, r1, l2, r2 = np.split(np.asarray(optimal_disc), axis=1, indices_or_sections=4)

        if np.min(l1) != float('-inf'):
            # Use left side
            negative_bounds.append([-self.max_bounds, np.min(l1)])
        if abs(np.max(l2) - np.min(r1)) > 1.0:
            # Use middle
            negative_bounds.append([np.min(r1), np.max(l2)])
        if np.max(r2) != float('inf'):
            # Use right side
            negative_bounds.append([np.max(r2), self.max_bounds])

        bnds = np.asarray(negative_bounds).flatten()
        if len(negative_bounds) == 0 or not np.array_equal(bnds, sorted(bnds)):
            raise Exception()

        positive_bounds = [[max(-self.max_bounds, np.min(l1)), np.min(r1)],
                           [np.max(l2), min(np.max(r2), self.max_bounds)]]
        positive_bounds = [x for x in positive_bounds if not math.isclose(x[0], x[1])]
        bnds = np.asarray(positive_bounds).flatten()
        if len(positive_bounds) == 0 or not np.array_equal(bnds, sorted(bnds)):
            raise Exception()

        for individual in populations[DISCRIMINATOR].individuals:
            if self.discriminator_collapse_type == [NEGATIVE, NEGATIVE]:
                bounds = random.choice(negative_bounds)
                l1 = random.uniform(bounds[0], bounds[1])
                r1 = random.uniform(max(l1, bounds[0]), bounds[1])

                secondary_allowed_bounds = negative_bounds[:negative_bounds.index(bounds) + 1]
                bounds = random.choice(secondary_allowed_bounds)
                l2 = random.uniform(max(r1, bounds[0]), bounds[1])
                r2 = random.uniform(max(l2, bounds[0]), bounds[1])

            elif self.discriminator_collapse_type == [POSITIVE, POSITIVE]:
                bounds = positive_bounds[0]
                l1 = random.uniform(bounds[0], bounds[1])
                r1 = random.uniform(max(l1, bounds[0]), bounds[1])

                bounds = positive_bounds[-1]
                l2 = random.uniform(max(r1, bounds[0]), bounds[1])
                r2 = random.uniform(max(l2, bounds[0]), bounds[1])

            elif self.discriminator_collapse_type == [NEGATIVE, POSITIVE] or \
                    self.discriminator_collapse_type == [POSITIVE, NEGATIVE]:
                if random.random() >= 0.5:
                    bounds = negative_bounds[0]
                    l1 = random.uniform(bounds[0], bounds[1])
                    r1 = random.uniform(max(l1, bounds[0]), bounds[1])

                    bounds = positive_bounds[-1]
                    l2 = random.uniform(max(r1, bounds[0]), bounds[1])
                    r2 = random.uniform(max(l2, bounds[0]), bounds[1])
                else:
                    bounds = positive_bounds[0]
                    l1 = random.uniform(bounds[0], bounds[1])
                    r1 = random.uniform(max(l1, bounds[0]), bounds[1])

                    bounds = negative_bounds[-1]
                    l2 = random.uniform(max(r1, bounds[0]), bounds[1])
                    r2 = random.uniform(max(l2, bounds[0]), bounds[1])

            else:
                raise Exception('Not supported')

            if not l1 < r1 < l2 < r2:
                # print(self.discriminator_collapse_type)
                raise Exception()

            genome = np.asarray([l1, r1, l2, r2])
            # inverse cumsum
            genome[1:] -= genome[:-1].copy()
            individual.genome = genome

    def _populate(self):
        populations = {}
        for population_name in POPULATION_NAMES:
            populations[population_name] = Population(individuals=[],
                                                      name=population_name,
                                                      sort_order=SORT_ORDERS[population_name],
                                                      default_fitness=DEFAULT_FITNESSES[population_name])
            for _ in range(self.population_size):
                genome = []
                if population_name == GENERATOR:
                    # Used for discriminator collapse
                    if isinstance(self.m1_init, list) and isinstance(self.m2_init, list):
                        m1 = random.uniform(self.m1_init[0], self.m1_init[1])
                        m2 = random.uniform(max(self.m2_init[0], m1), self.m2_init[1])
                        genome = np.asarray([m1, m2])
                        genome[1:] -= genome[:-1].copy()
                    # Used for mode collapse
                    elif self.m1_init is not None and self.m2_init is not None and \
                            not isinstance(self.m1_init, list) and not isinstance(self.m2_init, list):
                        genome = np.asarray([self.m1_init, self.m2_init])
                        genome[1:] -= genome[:-1].copy()

                    else:
                        # µ1 = [-BOUNDS, BOUNDS]
                        genome.append(random.uniform(-self.max_bounds, self.max_bounds))
                        # delta µ = [0, BOUNDS * 2] so that it is in [-BOUNDS, BOUNDS]
                        genome.append(random.uniform(0, self.max_bounds - genome[0]))
                else:
                    # l1 = [-BOUNDS, BOUNDS]
                    genome.append(random.uniform(-self.max_bounds, self.max_bounds - 3 * DISCRIMINATOR_INIT_DISTANCE))
                    # deltas 1,2,3 = [0, BOUNDS * 2] so that each is in [-BOUNDS, BOUNDS]
                    genome.append(random.uniform(0, DISCRIMINATOR_INIT_DISTANCE))
                    genome.append(random.uniform(0, DISCRIMINATOR_INIT_DISTANCE))
                    genome.append(random.uniform(0, DISCRIMINATOR_INIT_DISTANCE))

                solution = Individual(genome=genome, fitness=populations[population_name].default_fitness,
                                      population_size=self.population_size)
                populations[population_name].individuals.append(solution)
        return populations

    def mutate_gaussian(self, new_population, type):
        for i in range(len(new_population.individuals)):
            if random.random() < self.mutation_probability:
                genome = new_population.individuals[i].genome
                if type == GENERATOR:
                    # µ1 and µ2 (= µ1 + delta1) are clipped to [-BOUNDS, BOUNDS]
                    new_population.individuals[i].genome[0] = np.clip(genome[0] + self.step(), -self.max_bounds,
                                                                      self.max_bounds)
                    new_population.individuals[i].genome[1] = np.clip(genome[1] + self.step(), 0,
                                                                      self.max_bounds * 2 - genome[0])
                else:
                    # l1 is not clipped, i.e. [-inf, inf]
                    new_population.individuals[i].genome[0] += self.step()
                    # delta1, 2 and 3 are clipped to [0, inf] to ensure l1 < r1 < l2 < r2
                    new_population.individuals[i].genome[1] = max(0, genome[1] + self.step())
                    new_population.individuals[i].genome[2] = max(0, genome[2] + self.step())
                    new_population.individuals[i].genome[3] = max(0, genome[3] + self.step())

    def step(self):
        return random.gauss(0, 1) * self.gaussian_step

    def tournament_selection(self, population):
        assert 0 < self.tournament_size <= len(population.individuals), \
            "Invalid tournament size: {}".format(self.tournament_size)

        competition_population = Population(sort_order=population.sort_order,
                                            individuals=[],
                                            default_fitness=population.default_fitness,
                                            name='competition')
        new_population = Population(sort_order=population.sort_order,
                                    individuals=[],
                                    default_fitness=population.default_fitness,
                                    name=population.name)

        # Iterate until there are enough tournament winners selected
        while len(new_population.individuals) < self.population_size:
            # Randomly select tournament size individual solutions
            # from the population.
            competitors = random.sample(population.individuals, self.tournament_size)
            competition_population.individuals = competitors
            # Rank the selected solutions
            competition_population.sort_population()
            # Copy the solution
            winner = Individual(genome=list(competitors[0].genome),
                                fitness=competition_population.default_fitness, population_size=self.population_size)
            # Append the best solution to the winners
            new_population.individuals.append(winner)
        assert len(new_population.individuals) == self.population_size

        return new_population

    def evaluate_fitness(self, populations, fct):
        if self.evaluate_asymmetric:
            return self.evaluate_fitness_asym(populations, fct)
        return self.evaluate_fitness_sym(populations, fct)

    def evaluate_fitness_asym(self, populations, fct):
        for _, population in populations.items():
            for individual in population.individuals:
                individual.fitness = population.default_fitness

        for i, individual_generator in enumerate(populations[GENERATOR].individuals):

            k = [0.0] * len(populations[DISCRIMINATOR].individuals)

            for j, individual_discriminator in enumerate(populations[DISCRIMINATOR].individuals):

                k[j] = abs(fct(individual_generator, individual_discriminator, self.m1_opt, self.m2_opt,
                               self.max_bounds, self.optimal_discriminator))

                # Best worst case solution for generator
                if k[j] > individual_generator.fitness or \
                        individual_generator.fitness == populations[GENERATOR].default_fitness:
                    individual_generator.fitness = k[j]
                    individual_generator.adversary_solution = individual_discriminator.genome

                # Save all fitness values for logging
                populations[GENERATOR].individuals[i].all_fitnesses[j] = k[j]
                populations[DISCRIMINATOR].individuals[j].all_fitnesses[i] = k[j]

            # Sort discriminator population by k
            new_discriminator_individuals = [x for _, x in sorted(zip(k, populations[DISCRIMINATOR].individuals),
                                                                  key=lambda pair: pair[0])]

            for individual_discriminator in populations[DISCRIMINATOR].individuals:
                index = new_discriminator_individuals.index(individual_discriminator)

                if individual_discriminator.fitness == populations[DISCRIMINATOR].default_fitness:
                    individual_discriminator.fitness = index
                elif math.floor(individual_discriminator.fitness) != index:
                    individual_discriminator.fitness = max(individual_discriminator.fitness, index)
                else:
                    individual_discriminator.fitness = individual_discriminator.fitness + 1 / (self.population_size + 1)

    def evaluate_fitness_sym(self, populations, fct):
        for _, population in populations.items():
            for individual in population.individuals:
                individual.fitness = population.default_fitness

        for i in range(len(populations[DISCRIMINATOR].individuals)):
            for j in range(len(populations[GENERATOR].individuals)):

                fitness = abs(fct(populations[GENERATOR].individuals[j], populations[DISCRIMINATOR].individuals[i],
                                  self.m1_opt, self.m2_opt, self.max_bounds, self.optimal_discriminator))

                # Best worst case solution
                if fitness < populations[DISCRIMINATOR].individuals[i].fitness or \
                        populations[DISCRIMINATOR].individuals[i].fitness == populations[DISCRIMINATOR].default_fitness:
                    populations[DISCRIMINATOR].individuals[i].fitness = fitness
                    populations[DISCRIMINATOR].individuals[i].adversary_solution = populations[GENERATOR].individuals[
                        j].genome

                if fitness > populations[GENERATOR].individuals[j].fitness or \
                        populations[GENERATOR].individuals[j].fitness == populations[GENERATOR].default_fitness:
                    populations[GENERATOR].individuals[j].fitness = fitness
                    populations[GENERATOR].individuals[j].adversary_solution = populations[DISCRIMINATOR].individuals[
                        i].genome

                # Save all fitness values for logging
                populations[GENERATOR].individuals[j].all_fitnesses[i] = fitness
                populations[DISCRIMINATOR].individuals[i].all_fitnesses[j] = fitness


class CoevAlternating(Coev):

    def run(self):
        populations = self.initialize_populations()

        reverse_population_names = list(POPULATION_NAMES[:])
        reverse_population_names.reverse()

        t = 0
        self.evaluate_fitness(populations, self._fct)
        if self.verbose:
            for population_name in POPULATION_NAMES:
                print("t:{} {} best:{}".format(t, population_name, populations[population_name].individuals[0]))
            self.log_results(populations, t)

        t += 1

        while t < self.T:
            # Alternate between minimizer and maximizer populations
            for attacker, defender in (POPULATION_NAMES, reverse_population_names):
                if self.fix_discriminator and attacker == DISCRIMINATOR:
                    continue
                if self.fix_generator and attacker == GENERATOR:
                    continue

                if t >= self.T:
                    break

                new_population = self.tournament_selection(populations[attacker])
                self.mutate_gaussian(new_population, attacker)
                alternating_populations = {attacker: new_population,
                                           defender: populations[defender]}
                self.evaluate_fitness(alternating_populations, self._fct)

                # Replace the worst with the best new
                populations[attacker].replacement(new_population, self.n_replacements)
                # Print best
                populations[attacker].sort_population()
                if self.verbose:
                    print("run: {} t:{} {} best:{}".format(self.experiment_number, t, attacker,
                                                           populations[attacker].individuals[0]))
                    self.log_results(populations, t)

                t += 1

        return populations[DISCRIMINATOR].individuals[0].genome, \
               populations[DISCRIMINATOR].individuals[0].fitness, \
               populations[GENERATOR].individuals[0].genome, \
               populations[GENERATOR].individuals[0].fitness, \
               self.results


class CoevParallel(Coev):

    def run(self):
        populations = self.initialize_populations()

        t = 0

        self.evaluate_fitness(populations, self._fct)
        if self.verbose:
            for population_name in POPULATION_NAMES:
                print("t:{} {} best:{}".format(t, population_name, populations[population_name].individuals[0]))
            self.log_results(populations, t)

        t += 1

        while t < self.T:
            new_populations = {}

            if self.fix_discriminator:
                new_populations[DISCRIMINATOR] = populations[DISCRIMINATOR]
                keys = [GENERATOR]
            elif self.fix_generator:
                new_populations[GENERATOR] = populations[GENERATOR]
                keys = [DISCRIMINATOR]
            else:
                keys = populations.keys()

            for population_name in keys:
                new_populations[population_name] = self.tournament_selection(populations[population_name])
                self.mutate_gaussian(new_populations[population_name], population_name)

            self.evaluate_fitness(new_populations, self._fct)

            for population_name in keys:
                populations[population_name].replacement(new_populations[population_name], self.n_replacements)
                # Print best
                populations[population_name].sort_population()
                if self.verbose:
                    print("run: {} t:{} {} best:{}".format(self.experiment_number, t, population_name,
                                                           populations[population_name].individuals[0]))

            if self.verbose:
                self.log_results(populations, t)
            t += 1

        return populations[DISCRIMINATOR].individuals[0].genome, \
               populations[DISCRIMINATOR].individuals[0].fitness, \
               populations[GENERATOR].individuals[0].genome, \
               populations[GENERATOR].individuals[0].fitness, \
               self.results


def fct(generator, discriminator, m1_opt, m2_opt, max_bounds, optimal_discriminator):
    m1 = generator.genome[0]
    m2 = m1 + generator.genome[1]

    if optimal_discriminator:
        params = {'p_mu_1': m1_opt, 'p_mu_2': m2_opt, 'q_mu_1': m1, 'q_mu_2': m2}
        l1, r1, l2, r2 = find_optimal_discriminator(params)
        l1 = max(l1, -max_bounds)
        r2 = min(r2, max_bounds)

        # Inverse cumsum
        z = np.asarray([l1, r1, l2, r2])
        z[1:] -= z[:-1].copy()
        discriminator.genome = z
    else:
        bounds = np.cumsum(discriminator.genome)
        l1 = bounds[0]
        r1 = bounds[1]
        l2 = bounds[2]
        r2 = bounds[3]

    # Assuming that std=1
    return (0.5 * ((norm.cdf(r1, loc=m1_opt) - norm.cdf(l1, loc=m1_opt)) + (
            norm.cdf(r1, loc=m2_opt) - norm.cdf(l1, loc=m2_opt)) +
                   (norm.cdf(r2, loc=m1_opt) - norm.cdf(l2, loc=m1_opt)) + (
                           norm.cdf(r2, loc=m2_opt) - norm.cdf(l2, loc=m2_opt)))) - \
           (0.5 * ((norm.cdf(r1, loc=m1) - norm.cdf(l1, loc=m1)) + (norm.cdf(r1, loc=m2) - norm.cdf(l1, loc=m2)) +
                   (norm.cdf(r2, loc=m1) - norm.cdf(l2, loc=m1)) + (norm.cdf(r2, loc=m2) - norm.cdf(l2, loc=m2))))


# from outside: execute 100 times, append to exp file
def parallel_symmetric(params, experiment_number):
    alg = CoevParallel(fct, params['m1_opt'], params['m2_opt'], params['max_bounds'], params['gaussian_step'],
                       params['mutation_probability'], params['population_size'], params['tournament_size'],
                       params['n_replacements'], params['max_feval'], experiment_number + 1, params['verbose'],
                       evaluate_asymmetric=False, experiment_number=experiment_number)
    _, _, _, _, results = alg.run()
    return 'parallel_symmetric.csv', results


def parallel_asymmetric(params, experiment_number):
    alg = CoevParallel(fct, params['m1_opt'], params['m2_opt'], params['max_bounds'], params['gaussian_step'],
                       params['mutation_probability'], params['population_size'], params['tournament_size'],
                       params['n_replacements'], params['max_feval'], experiment_number + 2, params['verbose'],
                       evaluate_asymmetric=True, experiment_number=experiment_number)
    _, _, _, _, results = alg.run()
    return 'parallel_asymmetric.csv', results


def alternating_symmetric(params, experiment_number):
    alg = CoevAlternating(fct, params['m1_opt'], params['m2_opt'], params['max_bounds'], params['gaussian_step'],
                          params['mutation_probability'], params['population_size'], params['tournament_size'],
                          params['n_replacements'], params['max_feval'], experiment_number + 3, params['verbose'],
                          evaluate_asymmetric=False, experiment_number=experiment_number)
    _, _, _, _, results = alg.run()
    return 'alternating_symmetric.csv', results


def alternating_asymmetric(params, experiment_number):
    alg = CoevAlternating(fct, params['m1_opt'], params['m2_opt'], params['max_bounds'], params['gaussian_step'],
                          params['mutation_probability'], params['population_size'], params['tournament_size'],
                          params['n_replacements'], params['max_feval'], experiment_number + 4, params['verbose'],
                          evaluate_asymmetric=True, experiment_number=experiment_number)
    _, _, _, _, results = alg.run()
    return 'alternating_asymmetric.csv', results


def alternating_symmetric_opt_discriminator(params, experiment_number):
    alg = CoevAlternating(fct, params['m1_opt'], params['m2_opt'], params['max_bounds'], params['gaussian_step'],
                          params['mutation_probability'], params['population_size'], params['tournament_size'],
                          params['n_replacements'], params['max_feval'], experiment_number + 5, params['verbose'],
                          evaluate_asymmetric=False, optimal_discriminator=True, experiment_number=experiment_number)
    _, _, _, _, results = alg.run()
    return 'alternating_symmetric_opt_discr.csv', results


def alternating_asymmetric_opt_discriminator(params, experiment_number):
    alg = CoevAlternating(fct, params['m1_opt'], params['m2_opt'], params['max_bounds'], params['gaussian_step'],
                          params['mutation_probability'], params['population_size'], params['tournament_size'],
                          params['n_replacements'], params['max_feval'], experiment_number + 6, params['verbose'],
                          evaluate_asymmetric=True, optimal_discriminator=True, experiment_number=experiment_number)
    _, _, _, _, results = alg.run()
    return 'alternating_asymmetric_opt_discr.csv', results


def discriminator_collapse(params, experiment_number):
    bounds = params['max_bounds']

    combinations = [
        [NEGATIVE, NEGATIVE],
        [POSITIVE, NEGATIVE],
        [NEGATIVE, POSITIVE],
        [POSITIVE, POSITIVE]
    ]
    results = []
    for i, combination in enumerate(combinations):
        # m1_center = random.uniform(-(bounds - 1), bounds - 1)
        # m2_center = random.uniform(-(bounds - 1), bounds - 1)

        m1_center = -1
        m2_center = 2.5

        alg = CoevAlternating(fct, -2, 2, params['max_bounds'],
                              params['gaussian_step'], params['mutation_probability'],
                              params['population_size'], params['tournament_size'], params['n_replacements'],
                              params['max_feval'], experiment_number + 7 + 1000 + i, params['verbose'],
                              discriminator_collapse=True, experiment_number=experiment_number,
                              discriminator_collapse_type=combination, detailed_log=False,
                              m1_init=m1_center, m2_init=m2_center, fix_generator=True)
        _, _, _, _, intermediate_results = alg.run()
        results.extend(intermediate_results)

    return 'discriminator_collapse.csv', results


def mode_collapse(params, experiment_number, filename):
    eps = 1e-4 * (1.0 - -1.0)
    num = int((1.0 - -1.0) / (0.1 - eps) + 1)

    k = 0
    for m1 in np.linspace(1.0, -1.0, num):
        for m2 in np.linspace(-1.0, 1.0 - k, num - int(round(k * 10))):
            alg = CoevAlternating(fct, params['m1_opt'], params['m2_opt'], params['max_bounds'],
                                  params['gaussian_step'], params['mutation_probability'], params['population_size'],
                                  params['tournament_size'], params['n_replacements'], params['max_feval'],
                                  experiment_number + 8 + int(k * 10), params['verbose'], m1_init=m1, m2_init=m2,
                                  experiment_number=experiment_number, detailed_log=False)
            _, _, _, _, intermediate_results = alg.run()
            append_results_to_csv(filename, intermediate_results)

        k += 0.1


def append_results_to_csv(filename, data):
    path = 'experiment_results/' + filename
    first = not os.path.exists(path)
    keys = data[0].keys()
    with open(path, 'a') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        if first:
            dict_writer.writeheader()
        dict_writer.writerows(data)


def compute_n_times(n_chunk, function, params):
    result_array = []
    filename = None
    try:
        for i in n_chunk:
            filename, result = function(params, i)
            result_array.extend(result)
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise ex
    return filename, result_array, n_chunk


def compute_parallel(n_repeats, n_threads, function, params):
    result_array = [None] * n_threads
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(compute_n_times, n_chunk, function, params): i for i, n_chunk in
                   enumerate(np.array_split(range(n_repeats), n_threads))}
        for x in as_completed(futures):
            index = futures[x]
            filename, results, rng = x.result()
            result_array[index] = results
    end = time.time()
    print("Duration: {}".format(end - start))
    return filename, np.asarray(result_array).flatten()


def compute_mod_collapse_n_times(n_chunk, params):
    filename = 'temp_mod_collapse_{}.csv'.format(n_chunk[0])

    try:
        for i in n_chunk:
            mode_collapse(params, i, filename)
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise ex
    return filename


def compute_mode_collapse_parallel(n_repeats, n_threads, params):
    result_filenames = [None] * n_threads
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(compute_mod_collapse_n_times, n_chunk, params): i for i, n_chunk in
                   enumerate(np.array_split(range(n_repeats), n_threads))}
        for x in as_completed(futures):
            index = futures[x]
            result_filenames[index] = x.result()

    with open('experiment_results/mode_collapse.csv', 'w') as outfile:
        for fname in result_filenames:
            with open('experiment_results/' + fname) as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == '__main__':

    params = {
        'mutation_probability': 0.7,
        'population_size': 10,
        'tournament_size': 2,
        'n_replacements': 8,
        'gaussian_step': 1,
        'verbose': True,
        'seed': 1,
        'max_feval': 20000,
        'm1_opt': -2,
        'm2_opt': 2,
        'max_bounds': 10,
    }

    params_mod_collapse = dict(params)
    params_mod_collapse['m1_opt'] = -0.5
    params_mod_collapse['m2_opt'] = 0.5

    n_repeats = 120
    n_threads = 5

    filename, results = compute_parallel(n_repeats, n_threads, parallel_symmetric, params)
    append_results_to_csv(filename, results)
    
    filename, results = compute_parallel(n_repeats, n_threads, parallel_asymmetric, params)
    append_results_to_csv(filename, results)

    filename, results = compute_parallel(n_repeats, n_threads, alternating_symmetric, params)
    append_results_to_csv(filename, results)

    filename, results = compute_parallel(n_repeats, n_threads, alternating_asymmetric, params)
    append_results_to_csv(filename, results)

    filename, results = compute_parallel(n_repeats, n_threads, alternating_symmetric_opt_discriminator, params)
    append_results_to_csv(filename, results)

    filename, results = compute_parallel(n_repeats, n_threads, alternating_asymmetric_opt_discriminator, params)
    append_results_to_csv(filename, results)

    filename, results = compute_parallel(n_repeats, n_threads, discriminator_collapse, params_mod_collapse)
    append_results_to_csv(filename, results)

    compute_mode_collapse_parallel(n_repeats, n_threads, params_mod_collapse)
