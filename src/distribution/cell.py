from math import sqrt
from mpi4py import MPI

from distribution.codec import Codec

from constants import MASTER_RANK

world = MPI.COMM_WORLD
world_rank = world.Get_rank()
world_size = world.Get_size()


class Neighbourhood:
    def __init__(self, grid):

        self.rank = grid.Get_rank()

        n_rank, s_rank = grid.Shift(0, 1)
        w_rank, e_rank = grid.Shift(1, 1)

        self.south_rank = s_rank
        self.north_rank = n_rank
        self.west_rank = w_rank
        self.east_rank = e_rank

    @property
    def ranks(self):
        return [self.north_rank, self.south_rank, self.west_rank, self.east_rank]


class Population(dict):
    def __init__(self, neighbourhood, *args, **kwargs):
        super(Population, self).__init__(*args, **kwargs)
        self.neighbourhood = neighbourhood

    @property
    def center(self):
        return self[self.neighbourhood.rank]

    def sorted(self, key):
        return sorted(self.values(), key, reverse=True)

    @property
    def best(self, key):
        return self.sorted(key)[0]


class Cell:
    def __init__(self, grid, network_factory):
        dim = int(sqrt(world_size))
        self.grid: MPI.Cartcomm = world.Create_cart(dims=(dim, dim), periods=(1, 1))

        self.rank = grid.Get_rank()

        self.requests = []

        self.neighbourhood = Neighbourhood(grid)

        self.generators = Population(self.neighbourhood)
        self.discriminators = Population(self.neighbourhood)

        self.network_factory = network_factory

    @staticmethod
    def encode(G, D):
        return {
            "G": Codec.encode(G),
            "D": Codec.encode(D),
        }

    @staticmethod
    def decode(encoded, create_generator, create_discriminator):
        G = Codec.decode(encoded["G"], create_generator)
        D = Codec.decode(encoded["D"], create_discriminator)
        return G, D

    @property
    def encoded_center(self):
        return Cell.encode(self.generators.center, self.discriminators.center)

    def initialize(self):
        encoded = self.grid.neighbor_allgather(self.encoded_center)
        for i, rank in enumerate(self.neighbourhood.ranks):
            G, D = Cell.decode(
                encoded=encoded[i],
                create_generator=self.network_factory.create_generator(),
                create_discriminator=self.network_factory.create_discriminator(),
            )
            self.generators[rank] = G
            self.discriminators[rank] = D

    def finalize(self):
        return self.grid.gather(self.encoded_center, root=MASTER_RANK)

    def collect(self):
        status = MPI.Status()

        new_individuals = dict()
        while self.grid.iprobe(status=status):
            encoded = self.grid.recv()
            source = status.Get_source()

            new_individuals[source] = encoded

        for source, encoded in new_individuals.items():
            self[source] = self._decode_gan(encoded)

    def distribute(self):
        encoded = self.local_generator.encode()
        for neighbour_rank in self.neighbour_ranks:
            request = self.grid.isend(encoded, neighbour_rank)
            self.requests.append(request)
