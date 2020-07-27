"""
Author: Jamal Toutouh (toutouh@mit.edu) - www.jamal.es

This code is part of the research of our paper "Re-purposing Heterogeneous Generative Ensembles with Evolutionary
Computation" presented during GECCO 2020 (https://doi.org/10.1145/3377930.3390229)

evolutionary_nonrestricted_ensemble_optimization.py contains the code RestrictedEnsembleOptimization class, which
implements NREO-GEN optimization problem defined to create ensembles given the minimum and maximum size.
"""
import pathlib
import numpy as np
import glob
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class GAGeneratorIndividual(list):
    def __init__(self, attributes):
        # Some initialisation with received values
        self.attr1 = attributes
        pass


class NonRestrictedEnsembleOptimization:
    def __init__(
        self,
        min_ensemble_size=3,
        max_ensemble_size=8,
        generators_prefix="mnist-generator",
        generators_path="./mnist-generators/",
    ):
        precision = 100
        self.precision = precision
        self.generators_path = generators_path
        self.generators_prefix = generators_prefix
        self.generators_in_path = self.get_maximum_generators_index(generators_path, generators_prefix)
        self.min_ensemble_size = min_ensemble_size
        self.max_ensemble_size = (
            self.generators_in_path if max_ensemble_size > self.generators_in_path else max_ensemble_size
        )

    def mutate(self, individual):
        """It calls the mutation operator to be applied to an individual.
        :return: A tuple of one individual.
        """
        prob = random.random()
        if prob < 0.25:
            mutation_type = "just-weight-probability"
        elif prob < 0.50:
            mutation_type = "just-generator-index"
        elif prob < 0.75:
            mutation_type = "generator-and-weight"
        else:
            mutation_type = "just-ensemble-size"
        (individual,) = self.mutations(individual, mutation_type=mutation_type)
        return (individual,)

    def mutations(self, individual, mutation_type="generator-and-weight"):
        """Mutate an individual by replacing attributes.
        :param individual: :term:`Sequence <sequence>` individual to be mutated.
        :param mutation_type: Type of mutation to be applied (i.e., 'just-weight-probability', 'just-generator-index',
                              and 'generator-and-weight'
        :return: A tuple of one individual.
        """
        if mutation_type == "just-ensemble-size":
            ensemble_size = random.randint(self.min_ensemble_size, self.max_ensemble_size)
            individual = self.set_ensemble_size(individual, ensemble_size)
            individual = self.normalize_weights(individual)
        elif mutation_type == "just-weight-probability":
            weights = self.get_considered_weights(individual)
            i = random.randint(0, len(weights) - 1)
            weights[i] = random.randint(0, self.precision)
            self.set_indiviudal_weights(individual, weights)
            individual = self.normalize_weights(individual)
        elif mutation_type == "just-generator-index":
            generators = self.get_considered_generators(individual)
            if len(generators) < self.generators_in_path:
                i = random.randint(0, len(generators) - 1)
                new_idx = random.randint(len(generators) + 1, self.generators_in_path)
                individual = self.swap_genes(individual, i + 1, new_idx)
        elif mutation_type == "generator-and-weight":
            weights = self.get_considered_weights(individual)
            new_idx = -1
            if len(weights) < self.generators_in_path:
                new_idx = random.randint(len(weights) + 1, self.generators_in_path)
            i = random.randint(0, len(weights) - 1)
            weights[i] = random.randint(0, self.precision)
            if new_idx > 0:
                individual = self.swap_genes(individual, i + 1, new_idx)
            self.set_indiviudal_weights(individual, weights)
            individual = self.correct_solution(individual)
            individual = self.normalize_weights(individual)
        return (individual,)

    # ----------------- Auxiliary functions: Begin -----------------------------
    def get_individual_generators(self, ind):
        return ind[1 : self.generators_in_path + 1]

    def get_considered_generators(self, ind):
        return ind[1 : ind[0] + 1]

    def get_not_considered_generators(self, ind):
        return ind[ind[0] + 1 : self.generators_in_path + 1]

    def get_individual_weigths(self, ind):
        return ind[self.generators_in_path + 1 : 2 * self.generators_in_path + 1]

    def get_considered_weights(self, ind):
        return ind[self.generators_in_path + 1 : self.generators_in_path + 1 + ind[0]]

    def get_not_considered_weights(self, ind):
        return ind[self.generators_in_path + ind[0] + 1 : 2 * self.generators_in_path + 1]

    def set_indiviudal_weights(self, ind, weights):
        ind[self.generators_in_path + 1 : self.generators_in_path + 1 + len(weights)] = weights
        return

    # ----------------- Auxiliary functions: End -----------------------------

    def normalize_weights(self, ind):
        total_sum = sum(self.get_considered_weights(ind))
        for i in range(self.generators_in_path + 1, self.generators_in_path + ind[0]):
            ind[i] = int(self.precision * ind[i] / total_sum)
        ind[self.generators_in_path + ind[0]] = self.precision - sum(
            ind[self.generators_in_path + 1 : self.generators_in_path + ind[0]]
        )
        return ind

    def swap_genes(self, ind, idx1, idx2):
        aux = ind[idx1], ind[idx2]
        ind[idx2], ind[idx1] = aux
        return ind

    def cross_generators_genomes(self, ind1, ind2, cxpoint1, cxpoint2, offset):
        generators_ind1 = self.get_individual_generators(ind1)
        generators_ind2 = self.get_individual_generators(ind2)
        for gen in range(cxpoint1 + offset, cxpoint2 + offset):
            idx1 = generators_ind1.index(ind2[gen])
            idx2 = generators_ind2.index(ind1[gen])
            ind1 = self.swap_genes(ind1, gen, idx1 + offset)
            ind2 = self.swap_genes(ind2, gen, idx2 + offset)
        return ind1, ind2

    def show_mixture(self, ind):
        line = "Size = " + str(ind[0]) + " - Generator = ["
        for i in range(1, ind[0] + 1):
            line += str(int(ind[i])) + "-" + str(round(ind[self.generators_in_path + i] / self.precision, 2)) + ", "
        return line[:-2] + "]"

    def cross_weights_genomes(self, ind1, ind2, cxpoint1, cxpoint2, offset):
        cxpoint1 += offset
        cxpoint2 += offset
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
            ind2[cxpoint1:cxpoint2],
            ind1[cxpoint1:cxpoint2],
        )
        return ind1, ind2

    def cxTwoPointGAN(self, ind1, ind2):
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.
        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        size = max(ind1[0], ind2[0])
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Cross generators index --> Offset = 1
        ind1, ind2 = self.cross_generators_genomes(ind1, ind2, cxpoint1, cxpoint2, 1)

        # Cross weights index --> Offset = self.generators_in_path+1
        ind1, ind2 = self.cross_weights_genomes(ind1, ind2, cxpoint1, cxpoint2, self.generators_in_path + 1)
        ind1 = self.normalize_weights(ind1)
        ind2 = self.normalize_weights(ind2)

        return ind1, ind2

    def correct_solution(self, ind):  # Remove generators with weight 0
        weights = self.get_considered_weights(ind)
        offset = self.generators_in_path
        for i in range(len(weights)):
            if (
                weights[i] == 0
            ):  # Move generator and weight to the end of the ensemble and reduce the ensemble size by 1
                ind = self.swap_genes(ind, i + 1, ind[0])
                ind = self.swap_genes(ind, i + offset + 1, ind[0] + offset)
                ind[0] -= 1
        return ind

    def initialize_generators_index(self, max_generator_elements):
        """It randomly creates the generators the index part of the individual.
        :param max_generator_elements: Max number of generators.
        :return: The randomly shuffled generators index part of the individual.
        """
        generators_index = list(range(max_generator_elements))
        random.shuffle(generators_index)
        return generators_index

    def initialize_weights(self, max_generator_elements):
        """It randomly creates the generators the wights part of the individual.
        :param max_generator_elements: Max number of generators.
        :return: The randomly shuffled weights part of the individual.
        """
        return [random.randint(0, 10) for i in range(max_generator_elements)]

    def create_individual(self, individual_class):
        """It randomly creates individuals/solutions.
        :param individual_class: Class that defines the solutions.
        :return: An individual/solution.
        """
        ensemble_size = [random.randint(self.min_ensemble_size, self.max_ensemble_size)]
        generators_index = self.initialize_generators_index(self.generators_in_path)
        weights = self.initialize_weights(self.generators_in_path)
        individual = self.normalize_weights(ensemble_size + generators_index + weights)
        return individual_class(individual)

    def get_maximum_generators_index(self, generators_path, generators_prefix):
        """It gets the maximum number of generators to be used to create the ensembles.
        :parameter generators_path: Path where the generator files are stored.
        :parameter generators_prefix: Prefix of the file names that stores the generator model,
        :return: The maximum number of generators to be used to create the ensembles.
        """
        generators_found = len([gen for gen in glob.glob("{}*gene*.pkl".format(generators_path))])
        i = 0
        while len([gen for gen in glob.glob("{}/{}-{:03d}.pkl".format(generators_path, generators_prefix, i))]) > 0:
            i += 1
        if i == 0:
            print(
                "Error: No generators found in the path {} with the prefix {}".format(
                    generators_path, generators_prefix
                )
            )
            sys.exit(0)
        if generators_found != i:
            print(
                "Warning! {} found in the path {}, but the algorithm will use just {}. Check the genrators prefix.".format(
                    generators_found, generators_path, i
                )
            )
        return i

    def get_all_generators_in_path(self):
        return len([gen for gen in glob.glob("{}*gene*.pkl".format(self.generators_path))])

    def set_ensemble_size(self, individual, ensemble_size):
        individual[0] = ensemble_size
        return individual

    def get_generator_for_ensemble(self, generator_index):
        """It returns the full path of a given generator given its index.
        :param generator_index: Generator index.
        :return: Full path of a given generator given its index."""
        return (
            "{}/{}-{:03d}.pkl".format(self.generators_path, self.generators_prefix, int(generator_index),),
            "{:03d}".format(generator_index),
        )

    def get_mixture_from_individual(self, individual):
        tentative_weights = self.get_considered_weights(individual)
        generator_indices = self.get_considered_generators(individual)
        return (
            list(np.array(tentative_weights) / sum(tentative_weights)),
            [
                "{}/{}-{:03d}.pkl".format(self.generators_path, self.generators_prefix, int(generator_index),)
                for generator_index in generator_indices
            ],
            ["{:03d}".format(int(generator_index)) for generator_index in generator_indices],
        )

    def get_generators_for_ensemble(self, weight_and_generator_indices):
        """It returns the full path of a set of generators given the list of tuples that defines a solution/ensemble.
        :param weight_and_generator_indices: List that defines a solution/ensemble.
        :return: A tuple that contains the list of the full path of the generators and the indexes of the generators."""
        return (
            [
                "{}/{}-{:03d}.pkl".format(self.generators_path, self.generators_prefix, int(generator_index),)
                for weight, generator_index in weight_and_generator_indices
            ],
            ["{:03d}".format(int(generator_index)) for weight, generator_index in weight_and_generator_indices],
        )

    @staticmethod
    def extract_weights(weight_and_generator_indices):
        """It extracts the weights of a solution to be printed.
        :param weight_and_generator_indices: Individual/solution that defines an ensemble.
        :return: List of weights that defines the mixture."""
        return [round(weight, 1) for weight, generator_index in weight_and_generator_indices]

    def show_ensemble_size_info(self):
        return "Min ensemble size={}, Max ensemble size={}, ".format(self.min_ensemble_size, self.max_ensemble_size)
