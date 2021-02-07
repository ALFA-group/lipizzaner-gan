from math import sqrt
from collections import OrderedDict

import numpy as np

from distribution.client_environment import ClientEnvironment
from distribution.concurrent_populations import ConcurrentPopulations
from distribution.node_client import NodeClient
from helpers.configuration_container import ConfigurationContainer
from helpers.math_helpers import is_square
from helpers.network_helpers import is_local_host
from helpers.population import Population, TYPE_GENERATOR, TYPE_DISCRIMINATOR
from helpers.singleton import Singleton


@Singleton
class Neighbourhood:
    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.concurrent_populations = ConcurrentPopulations.instance()

        dataloader = self.cc.create_instance(self.cc.settings["dataloader"]["dataset_name"])
        network_factory = self.cc.create_instance(
            self.cc.settings["network"]["name"], dataloader.n_input_neurons, num_classes=dataloader.num_classes,
        )
        self.node_client = NodeClient(network_factory)

        (self.grid_size, self.grid_position, self.local_node,) = self._load_topology_details()
        self.cell_number = self._load_cell_number()

        num_clients = self.cc.settings["general"]["distribution"].get("num_clients", None)

        self.scheduler = self.cc.settings["trainer"]["params"]["fitness"].get("scheduler", None)
        if self.scheduler is not None:
            self.alpha= self.scheduler['0']['alpha']
            self.beta = self.scheduler['0']['beta']
        else:
            self.alpha = self.cc.settings["trainer"]["params"]["fitness"].get("alpha", None)
            self.beta = self.cc.settings["trainer"]["params"]["fitness"].get("beta", None)
            if num_clients is not None and (self.alpha is None or self.beta is None):
                if num_clients == 1:
                    self.alpha = 0.5
                    self.beta = 0.5
                else:
                    self.alpha = self.cell_number / (num_clients - 1)
                    self.beta = 1 - self.alpha

        self.neighbours = self._adjacent_cells()
        self.all_nodes = self.neighbours + [self.local_node]

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

    def _load_topology_details(self):
        client_nodes = self._all_nodes_on_grid()

        if len(client_nodes) != 1 and not is_square(len(client_nodes)):
            raise Exception("Provide either one client node, or a square number of cells (to create a square grid).")

        local_port = ClientEnvironment.port
        matching_nodes = [
            node for node in client_nodes if is_local_host(node["address"]) and int(node["port"]) == local_port
        ]

        if len(matching_nodes) == 1:
            dim = int(round(sqrt(len(client_nodes))))
            idx = client_nodes.index(matching_nodes[0])
            x = idx % dim
            y = idx // dim
            return len(client_nodes), (x, y), matching_nodes[0]
        else:
            raise Exception(
                "This host is not specified as client in the configuration file, "
                "or too many clients match the condition."
            )

    def _load_cell_number(self):
        x, y = self.grid_position
        return y * int(sqrt(self.grid_size)) + x

    def _adjacent_cells(self):
        if self.grid_size == 1:
            return []

        nodes = self._all_nodes_on_grid()
        for node in nodes:
            node["id"] = "{}:{}".format(node["address"], node["port"])

        dim = int(round(sqrt(len(nodes))))
        x, y = self.grid_position
        nodes = np.reshape(nodes, (-1, dim))

        def neighbours(x, y):
            indices = np.array([(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)])
            # Start at 0 when x or y is out of bounds
            indices[indices >= dim] = 0
            indices[indices == -1] = dim - 1
            # Remove duplicates (needed for smaller grids), and convert to (x,y) tuples
            return np.array([tuple(row) for row in np.unique(indices, axis=0)])

        mask = np.zeros((dim, dim))
        mask[tuple(neighbours(x, y).T)] = 1

        return nodes[mask == 1].tolist()

    def _all_nodes_on_grid(self):
        nodes = self.cc.settings["general"]["distribution"]["client_nodes"]
        for node in nodes:
            node["id"] = "{}:{}".format(node["address"], node["port"])
        return nodes

    def _set_source(self, population):
        for individual in population.individuals:
            individual.source = "{}:{}".format(self.local_node["address"], self.local_node["port"])
        return population

    def _init_mixture_weights(self):
        node_ids = [node["id"] for node in self.all_nodes]
        default_weight = 1 / len(node_ids)
        # Warning: Feature of order preservation in Dict is used in the mixture_weight
        #          initialized here because further code involves converting it to list
        # According to https://stackoverflow.com/a/39980548, it's still preferable/safer
        # to use OrderedDict over Dict in Python 3.6
        return OrderedDict({n_id: default_weight for n_id in node_ids})
