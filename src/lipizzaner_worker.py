import torch
from mpi4py import MPI

from constants import MASTER_RANK

from helpers.configuration_container import ConfigurationContainer
from helpers.reproducible_helpers import set_random_seed

from distribution.cell import Cell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

world = MPI.COMM_WORLD
world_rank = world.Get_rank()
world_size = world.Get_size()


class LipizzanerWorker:
    def __init__(self, trainer=None):

        self.is_master = world_rank == MASTER_RANK
        self._configure_trainer(trainer)
        self.cell = Cell()

        # It is not possible to obtain reproducible result for large grid due to nature of asynchronous training
        # But still set seed here to minimize variance
        set_random_seed(self.rank, self.cuda)

    @property
    def neighbour_ranks(self):
        return [self.south_rank, self.north_rank, self.east_rank, self.west_rank]

    def _configure_trainer(self, trainer):
        if trainer is not None:
            self.trainer = trainer
        else:
            self.cc = ConfigurationContainer.instance()
            dataloader = self.cc.create_instance(self.cc.settings["dataloader"]["dataset_name"])
            network_factory = self.cc.create_instance(
                self.cc.settings["network"]["name"],
                dataloader.n_input_neurons,
                num_classes=dataloader.num_classes,
            )
            self.trainer = self.cc.create_instance(
                self.cc.settings["trainer"]["name"],
                dataloader,
                network_factory,
            )

        if "params" in self.cc.settings["trainer"] and "score" in self.cc.settings["trainer"]["params"]:
            self.cuda = self.cc.settings["trainer"]["params"]["score"]["cuda"]
        else:
            self.cuda = False

    def run(self, n_iterations):
        print(f"Running {self.__class__.__name__} {world_rank} (or {self.rank} in grid)")

        self.cell.initialize()

        self.trainer._prepare_data()

        for iteration in range(n_iterations):
            self.cell.collect()
            self.trainer._run_generation(iteration)
            self.cell.distribute()

        gans = self.cell.finalize()

        if self.is_master:
            self.trainer._optimize_generator_mixture(gans)

        self.grid.Barrier()
