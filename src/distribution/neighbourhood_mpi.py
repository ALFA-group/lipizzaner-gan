import math
from collections import OrderedDict
from math import sqrt

import numpy as np
from helpers.configuration_container import ConfigurationContainer
from helpers.math_helpers import is_square
from helpers.network_helpers import is_local_host
from helpers.population import TYPE_DISCRIMINATOR, TYPE_GENERATOR, Population
from helpers.singleton import Singleton
from mpi4py import MPI

from distribution.client_environment import ClientEnvironment
from distribution.concurrent_populations import ConcurrentPopulations
from distribution.node_client import NodeClient

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

NEIGHBOUR_DIRS =  [(0,1),(0,-1),(1,0),(-1,0)]


@Singleton
class Neighbourhood:
    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.concurrent_populations = ConcurrentPopulations.instance()

        dataloader = self.cc.create_instance(self.cc.settings["dataloader"]["dataset_name"])
        network_factory = self.cc.create_instance(
            self.cc.settings["network"]["name"], dataloader.n_input_neurons, num_classes=dataloader.num_classes,
        )

        self.cell_number = rank
        self.grid_size = sqrt(size)
        self.grid_position =  (rank // self.grid_size, rank % self.grid_size)


        self.scheduler = self.cc.settings["trainer"]["params"]["fitness"].get("scheduler", None)
        if self.scheduler is not None:
            self.alpha= self.scheduler['0']['alpha']
            self.beta = self.scheduler['0']['beta']
        else:
            self.alpha = self.cc.settings["trainer"]["params"]["fitness"].get("alpha", None)
            self.beta = self.cc.settings["trainer"]["params"]["fitness"].get("beta", None)
            if size is not None and (self.alpha is None or self.beta is None):
                if size == 1:
                    self.alpha = 0.5
                    self.beta = 0.5
                else:
                    self.alpha = self.cell_number / (size - 1)
                    self.beta = 1 - self.alpha

        self.neighbour_coords = [
            ((self.coord + i) % self.grid_size, (self.coord + j) % self.grid_size) 
            for i,j in NEIGHBOUR_DIRS
        ]

        self.neighbours = [self._coord_to_rank(coord) for coord in self.neighbour_coords]

        self.all_nodes = self.neighbours + [self.cell_number]

        self.mixture_weights_generators = self._init_mixture_weights()
        if (
            self.cc.settings["trainer"]["name"] == "with_disc_mixture_wgan"
            or self.cc.settings["trainer"]["name"] == "with_disc_mixture_gan"
        ):
            self.mixture_weights_discriminators = self._init_mixture_weights()
        else:
            self.mixture_weights_discriminators = None

    @property
    def local_generators(self):
        # Return local individuals for now, possibility to split up gens and discs later
        return self._set_source(self.concurrent_populations.generator)

    @property
    def local_discriminators(self):
        # Return local individuals for now, possibility to split up gens and discs later
        return self._set_source(self.concurrent_populations.discriminator)

    @property
    def all_generators(self):
        neighbour_individuals = self.node_client.get_all_generators(self.neighbours)
        local_population = self.local_generators

        return Population(
            individuals=neighbour_individuals + local_population.individuals,
            default_fitness=local_population.default_fitness,
            population_type=TYPE_GENERATOR,
        )

    @property
    def best_generators(self):
        best_neighbour_individuals = self.node_client.get_best_generators(self.neighbours)
        local_population = self.local_generators
        best_local_individual = sorted(local_population.individuals, key=lambda x: x.fitness)[0]

        return Population(
            individuals=best_neighbour_individuals + [best_local_individual],
            default_fitness=local_population.default_fitness,
            population_type=TYPE_GENERATOR,
        )

    @property
    def all_discriminators(self):
        neighbour_individuals = self.node_client.get_all_discriminators(self.neighbours)
        local_population = self.local_discriminators

        return Population(
            individuals=neighbour_individuals + local_population.individuals,
            default_fitness=local_population.default_fitness,
            population_type=TYPE_DISCRIMINATOR,
        )

    @property
    def all_generator_parameters(self):
        neighbour_generators = self.node_client.load_generators_from_api(self.neighbours)
        local_parameters = [i.genome.encoded_parameters for i in self.local_generators.individuals]
        return local_parameters + [n["parameters"] for n in neighbour_generators]

    @property
    def all_discriminator_parameters(self):
        neighbour_discriminators = self.node_client.load_discriminators_from_api(self.neighbours)
        local_parameters = [i.genome.encoded_parameters for i in self.local_discriminators.individuals]
        return local_parameters + [n["parameters"] for n in neighbour_discriminators]

    @property
    def best_generator_parameters(self):
        return self.node_client.load_best_generators_from_api(self.neighbours + [self.local_node])

    @property
    def best_discriminator_parameters(self):
        return self.node_client.load_best_discriminators_from_api(self.neighbours + [self.local_node])

    def _set_source(self, population):
        for individual in population.individuals:
            individual.source = self.grid_position
        return population

    def _init_mixture_weights(self):
        node_ids = [node["id"] for node in self.all_nodes]
        default_weight = 1 / len(node_ids)
        # Warning: Feature of order preservation in Dict is used in the mixture_weight
        #          initialized here because further code involves converting it to list
        # According to https://stackoverflow.com/a/39980548, it's still preferable/safer
        # to use OrderedDict over Dict in Python 3.6
        return OrderedDict({n_id: default_weight for n_id in node_ids})

    def _coord_to_rank(self, coord):
        i,j = coord
        return self.grid_size * i + j
