import numpy as np
from itertools import product
import glob
import random
from deap import base
from deap import creator
from deap import tools
import math
import random
from itertools import repeat
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class TVDBaserGA:
    def __init__(self, precision=10, generators_path='./generators/', mode='iterative'):
        self.precision = precision
        self.possible_weights = self.create_weights_list(0, self.precision+1, 1)
        self.generators_path = generators_path
        self.generators_in_path = self.get_all_generators_in_path()



    def get_all_generators_in_path(self):
        return len([gen for gen in glob.glob('{}*gene*.pkl'.format(self.generators_path))])

    def create_weights_list(self, min_value, max_value, step_size):
        return np.arange(min_value, max_value, step_size)

    def get_genarators_for_ensemble(self, size):
        generator_indizes = random.sample(range(0, self.generators_in_path), size)
        return ['{}mnist-generator-{:03d}.pkl'.format(self.generators_path, generator_index) for generator_index in generator_indizes], \
               ['{:03d}'.format(generator_index) for generator_index in generator_indizes]

    def get_possible_tentative_weights(self, current_prob, i, size):
        max_prob = 1 * self.precision
        number_of_following_weights = size - i
        if number_of_following_weights==1:
            return [max_prob - current_prob]
        else:
            max_probability = (max_prob - current_prob) / number_of_following_weights
            return self.possible_weights[self.possible_weights <= max_probability]

    def get_weights_tentative(self, size):
        result = []
        for i in range(size):
            current_prob = sum(result)
            possible_weights = self.get_possible_tentative_weights(current_prob, i, size)
            result.append(random.choice(possible_weights))
            i += 1
        return np.array(result) / self.precision, len(result)


ga = TVDBaserGA()

max_generators_index = 10
ensemble_size = 5

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
#toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("attr_rand", random.uniform, 0, max_generators_index)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rand, ensemble_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def mutations(individual, low, up, indpb, mutation_type='generator-and-weight'):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    if mutation_type == 'just-weight-probabilty':
        for i, xl, xu in zip(range(size), low, up):
            if random.random() < indpb:
                individual[i] = random.uniform(xl, xu)
    elif mutation_type == 'just-generator-index':
        print('Indiv: {}'.format(individual))
        for i, xl, xu in zip(range(size), low, up):
            if random.random() < indpb:
                individual[i] = abs(individual[i] + random.randint(xl, xu-1) - xu)
                print('MUTATING: Indiv: {}'.format(individual))
    elif mutation_type == 'generator-and-weight':
        print('Indiv: {}'.format(individual))
        for i, xl, xu in zip(range(size), low, up):
            if random.random() < indpb:
                individual[i] = random.uniform(xl, xu)
                print('MUTATING: Indiv: {}'.format(individual))
    print('Indiv: {}'.format(individual))
    return individual,


def create_individual(individual):
    individual = [float(round(gen)) for gen in individual]
    print([ind for ind in individual])
    weights = ga.get_weights_tentative(len(individual))[0]
    print([ind for ind in weights])

    for i, weight in zip(range(len(individual)), weights):
        individual[i] += float(weight)
    print([ind for ind in individual])
    return individual,


def mutate(individual):
    print('Indiv: {}'.format(individual))
    prob = 0.5 #random.random()
    if prob < 0.33:
        mutation_type ='just-weight-probabilty'
    elif prob < 0.6:
        mutation_type = 'just-generator-index'
    else:
        mutation_type = 'generator-and-weight'
    print('Mutation {}: {}'.format(mutation_type, individual))
    individual, = toolbox.mutations(individual, mutation_type=mutation_type)

    return individual,



def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("mutations", mutations, low=0, up=max_generators_index, indpb=10/ensemble_size, mutation_type='just-generator-index')
# toolbox.register("mutate_generator", tools.mutUniformInt,  low=0, up=max_generators_index, indpb=1/ensemble_size)
# toolbox.register("mutate_probability", tools.mutUniformInt,  low=0, up=max_generators_index, indpb=1/ensemble_size)
# toolbox.register("mutate_both_generator_probability", mutUniformFloat,  low=0, up=max_generators_index, indpb=1/ensemble_size)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=3)
    print([ind for ind in pop])

    #pop = map(create_individual, pop)
    #print([ind for ind in pop])
    offspring = tools.selBest(pop, len(pop))
    for i in range(len(offspring)):
        offspring[i] = create_individual(offspring[i])
    print(offspring)
    pop[:   ] = offspring

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 1000000 and g < 10:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                print(mutant)
                toolbox.mutate(mutant)
                del mutant.fitness.values
                print(mutant)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

main()

