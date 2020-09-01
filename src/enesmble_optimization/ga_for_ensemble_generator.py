"""
Author: Jamal Toutouh (toutouh@mit.edu) - www.jamal.es

This code is part of the research of our paper "Re-purposing Heterogeneous Generative Ensembles with Evolutionary
Computation" presented during GECCO 2020 (https://doi.org/10.1145/3377930.3390229)

ga_for_ensemble_generator.py contains the code to create ensembles of (GAN) generators by using evolutionary computing.
The GAEnsembleGenerator class uses one of two different classes:
- RestrictedEnsembleOptimization class: to create ensembles with a specific size, i.e., REO-GEN.
- NonRestrictedEnsembleOptimization class: to create ensembles given the minimum and maximum size, i.e., NREO-GEN.
"""

import pathlib
import sys
import time
from deap import base
from deap import creator
from deap import tools
import random
import importlib
from scipy.stats import iqr
import torch
import numpy as np
import glob

sys.path.insert(
    0, str(pathlib.Path(__file__).parent.parent.absolute())
)  # To change the folder path to use Lipizzaner files
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.population import Population
from training.mixture.score_factory import ScoreCalculatorFactory
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset


try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class GAEnsembleGenerator:

    fitness_types = [
        "tvd",
        "fid",
        "tvd-fid",
    ]  # Currently supported GAN metrics. 'tvd-fid' for future multi-objective
    class_maps = {
        "reo-gen": (
            "enesmble_optimization.evolutionary_restricted_ensemble_optimization",
            "RestrictedEnsembleOptimization",
        ),
        "nreo-gen": (
            "enesmble_optimization.evolutionary_nonrestricted_ensemble_optimization",
            "NonRestrictedEnsembleOptimization",
        ),
    }

    def __init__(
        self,
        dataset,
        min_ensemble_size,
        max_ensemble_size,
        generators_path,
        generators_prefix,
        fitness_metric,
        evolutionary_approach,
        population_size,
        number_of_generations,
        number_of_fitness_evaluations,
        mutation_probability,
        crossover_probability,
        show_info_iteration,
        output_file="",
    ):
        self.ga = self.create_instance(
            evolutionary_approach, min_ensemble_size, max_ensemble_size, generators_prefix, generators_path,
        )  # TVDBasedGA()
        self.store_output_file = output_file
        self.show_info_iteration = show_info_iteration

        # Configure the parameters required to use Lipizzaner to evaluate the ensemble
        self.configure_lipizzaner(dataset)

        # Configure the parameters required of the EA
        self.configure_evolutionary_algorithm(
            population_size,
            crossover_probability,
            mutation_probability,
            fitness_metric,
            number_of_generations,
            number_of_fitness_evaluations,
        )

    def configure_evolutionary_algorithm(
        self,
        population_size,
        crossover_probability,
        mutation_probability,
        fitness_metric,
        number_of_generations,
        number_of_fitness_evaluations,
    ):
        """It configures the parameters required for the EA.
        :parameter population_size: Size of the population
        :parameter crossover_probability: Crossover probability
        :parameter mutation_probability: Mutation probability
        :parameter fitness_metric: Metric used to evaluate the quality of the ensemble
        :parameter number_of_generations: Maximum number of the evolutionary steps (stop condition)
        :parameter number_of_fitness_evaluations: Maximum number of fitness evaluations (stop condition)
        """
        fitness_type = fitness_metric if fitness_metric in self.fitness_types else "tvd"
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        # self.toolbox.register("attr_rand", random.uniform, 0, max_generators_index)
        self.toolbox.register("individual", self.ga.create_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register(
            "evaluate",
            self.evaluate_ensemble,
            network_factory=self.network_factory,
            mixture_generator_samples_mode=self.mixture_generator_samples_mode,
            fitness_type=fitness_type,
        )
        self.toolbox.register("mutate", self.ga.mutate)
        self.toolbox.register("crossoverGAN", self.ga.cxTwoPointGAN)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.pop = self.toolbox.population(n=population_size)
        self.CXPB, self.MUTPB = crossover_probability, mutation_probability
        self.number_of_generations = number_of_generations
        self.number_of_fitness_evaluations = number_of_fitness_evaluations
        self.fitness_type = fitness_type

    def configure_lipizzaner(self, dataset):
        """It configures the parameters required to use Lipizzaner to evaluate the ensembles.
        :parameter dataset: dataset addressed in the problem
        """
        cuda_availability = torch.cuda.is_available()
        cc = ConfigurationContainer.instance()
        settings = {
            "trainer": {"params": {"score": {"type": "fid"}}},
            "dataloader": {"dataset_name": dataset},
            "master": {"cuda": cuda_availability},
            "network": {"loss": "bceloss", "name": "four_layer_perceptron"},
            "general": {"distribution": {"auto_discover": "False"}, "output_dir": "./output", "num_workers": 0,},
        }
        cc.settings = settings
        data_loader = cc.create_instance(dataset)
        self.mixture_generator_samples_mode = "exact_proportion"
        self.score_calc = ScoreCalculatorFactory.create()
        self.network_factory = cc.create_instance("four_layer_perceptron", data_loader.n_input_neurons)

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

    def create_instance(self, name, *args):
        """It instances the class that contains the EA
        :parameter name: Name of the class
        :parameter *args: Arguments to create the class
        :return: An instance of the EA class
        """
        module_name, class_name = self.class_maps[name]
        cls = getattr(importlib.import_module(module_name), class_name)
        if name == "reo-gen":
            args = (args[0], args[2], args[3])
        return cls(*args)

    def evaluate_ensemble(
        self, individual, network_factory, mixture_generator_samples_mode="exact_proportion", fitness_type="tvd",
    ):
        """It evaluates the solution/individual (ensemble) given the fitness type. It generates samples and it evaluates
        the metric defined by fitness_type using Lipizzaner.
        :parameter individual: Solutionto be evaluated
        :parameter network_factory:
        :parameter mixture_generator_samples_mode:
        :parameter fitness_type: It defines the type of metric to be evaluated.
        :return: The fitness_type metric value got by the solution.
        """
        population = Population(individuals=[], default_fitness=0)
        # weight_and_generator_indices = [math.modf(gen) for gen in individual]
        # generators_paths, sources = self.ga.get_generators_for_ensemble(weight_and_generator_indices)
        # tentative_weights = [weight for weight, generator_index in weight_and_generator_indices]
        (tentative_weights, generators_paths, sources,) = self.ga.get_mixture_from_individual(individual)
        mixture_definition = dict(zip(sources, tentative_weights))
        for path, source in zip(generators_paths, sources):
            generator = network_factory.create_generator()
            generator.net.load_state_dict(torch.load(path, map_location="cpu"))
            generator.net.eval()
            population.individuals.append(Individual(genome=generator, fitness=0, source=source))
        dataset = MixedGeneratorDataset(population, mixture_definition, 50000, mixture_generator_samples_mode,)
        fid, tvd = self.score_calc.calculate(dataset)

        if fitness_type == "tvd":
            return (tvd,)
        elif fitness_type == "fid":
            return (fid,)
        elif fitness_type == "tvd-fid":
            return ((tvd, fid),)

    def evaluate_tvd_fid(self, ind):
        """It returns the tvd and fid of a given solution (i.e., ensemble)
        :parameter ind: A given solution/individual.
        :return tvd: TVD
        :return fid: FID
        """
        result = self.evaluate_ensemble(ind, self.network_factory, fitness_type="tvd-fid")
        tvd = result[0][0]
        fid = result[0][1]
        return tvd, fid

    def get_fitness_stats(self, fitness):
        """It returns the stats of the a list of fitness values.
        :parameter fitness: List of fitness values.
        :return stats: Dictionary that stores the different statistical metrics.
        """
        stats = dict()
        num = np.array(fitness)
        stats["iqr"] = iqr(num)
        stats["mean"] = num.mean()
        stats["median"] = np.median(num)
        stats["max"] = num.max()
        stats["min"] = num.min()
        stats["count"] = len(num)
        stats["stdev"] = num.std()
        stats["norm_stdev"] = stats["stdev"] / stats["mean"]
        return stats

    def show_population_tvd_fid(self, output_file=""):
        """It shows the tvd and fid of the individuals of the whole population.
        """
        text = ""
        for ind in self.pop:
            tvd, fid = self.evaluate_tvd_fid(ind)
            text += "{} - TVD={}, FID={} \n".format(self.ga.show_mixture(ind), tvd, fid)
        self.show_file_screen(text, output_file)

    def show_evolution_stats(self, generations, fitness_evaluation, output_file=""):
        """It gathers all the fitness values and shows the main stats.
        :parameter generations: Current generation
        :parameter fitness_evaluation: Current number of fitness evaluations
        """
        fitness_info = [ind.fitness.values[0] for ind in self.pop]
        stats = self.get_fitness_stats(fitness_info)
        text = (
            "Fitness Information: Generation={}, Fitness evaluations={}, Population size={}, Mean={}, Stdev={}, "
            "Norm Stedv={}, Median={}, IQR={}, Min={}, Max={}".format(
                generations,
                fitness_evaluation,
                stats["count"],
                stats["mean"],
                stats["stdev"],
                stats["norm_stdev"],
                stats["median"],
                stats["iqr"],
                stats["min"],
                stats["max"],
            )
        )
        self.show_file_screen(text, output_file)

    def show_population_info(self, generations, fitness_evaluation, output_file=""):
        """It shows the population genomes and their fitness
        :parameter generations: Current generation
        :parameter fitness_evaluation: Current number of fitness evaluations
        """
        pop_info = [[ind, ind.fitness.values[0]] for ind in self.pop]

        text = "-Generation: {} \t-Fitness evaluations: {}\n".format(generations, fitness_evaluation)
        for ind, fitness in pop_info:
            text += "{} - {} \n".format(self.ga.show_mixture(ind), fitness)
        self.show_file_screen(text, output_file)

    def show_evolutionary_algorithm_configuration(self, output_file=""):
        config_info = "EA configuration: "
        config_info += "Fitness function={}, ".format(self.fitness_type)
        config_info += "Number of generations={}, ".format(self.number_of_generations)
        config_info += "{}".format(self.ga.show_ensemble_size_info())
        config_info += "Fitness evaluations={}, ".format(self.number_of_fitness_evaluations)
        config_info += "Crossover probability={}, ".format(self.CXPB)
        config_info += "Mutation probability={}, ".format(self.MUTPB)
        config_info += "Output file={} ".format(self.store_output_file)
        self.show_file_screen(config_info, output_file)

    def show_file_screen(self, text, output_file=""):
        if output_file == "":
            print(text)
        else:
            file = open(output_file, "a")
            file.write(text)
            file.close()

    def evolutionary_loop(self):
        """It defines the evolutionary process to be carried out by the EA.
        :return pop: Population
        :return stats: Fitness stats
        :return computation_time: Computation time in seconds
        """
        self.show_evolutionary_algorithm_configuration()
        if self.store_output_file != "":
            self.show_evolutionary_algorithm_configuration(self.store_output_file)

        init_time = time.clock()

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        self.show_population_tvd_fid()
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        # Variables keeping track of the number of generations and fitness evaluations
        generation = 0
        fitnesses_evaluation = 0

        print("Initial population:")
        self.show_population_tvd_fid()
        self.show_evolution_stats(generation, fitnesses_evaluation)

        counter = 0
        if self.number_of_fitness_evaluations > 0:
            stop_condition = self.number_of_fitness_evaluations
        elif self.number_of_generations > 0:
            stop_condition = self.number_of_generations
        else:
            print(
                "Error: No stop condition set: Please configure the Number of fitness evaluations or the Number of "
                "generations"
            )
            sys.exit(0)

        # Begin the evolution
        while counter < stop_condition:
            # A new generation
            generation += 1
            print("Generation: {}".format(generation))

            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    self.toolbox.crossoverGAN(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fitnesses_evaluation += len(invalid_ind)
            self.pop[:] = offspring

            if self.show_info_iteration != 0 and generation % self.show_info_iteration == 0:
                self.show_population_info(generation, fitnesses_evaluation)
                self.show_evolution_stats(generation, fitnesses_evaluation)

            if self.store_output_file != "":
                self.show_population_info(generation, fitnesses_evaluation, self.store_output_file)
                self.show_evolution_stats(generation, fitnesses_evaluation, self.store_output_file)

            counter = generation if self.number_of_generations > 0 else fitnesses_evaluation

        computation_time = time.clock() - init_time
        print("Final population:")
        self.show_evolution_stats(generation, fitnesses_evaluation)
        self.show_population_tvd_fid()
        print("Computation time: {}".format(computation_time))

        if self.store_output_file != "":
            self.show_file_screen(
                "Computation time: {}".format(computation_time), self.store_output_file,
            )
            self.show_file_screen("Final population:", self.store_output_file)
            self.show_evolution_stats(generation, fitnesses_evaluation, self.store_output_file)
            self.show_population_tvd_fid(self.store_output_file)
        return (
            self.pop,
            self.get_fitness_stats(ind.fitness.values),
            computation_time,
        )
