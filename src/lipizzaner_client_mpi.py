import logging
import os
import traceback
from threading import Event, Lock, Thread

import torch
import torch.utils.data
from mpi4py import MPI
from torch.autograd import Variable

from distribution.concurrent_populations import ConcurrentPopulations
from helpers.configuration_container import ConfigurationContainer
from helpers.log_helper import LogHelper
from helpers.or_event import or_event
from helpers.reproducible_helpers import set_random_seed
from lipizzaner import Lipizzaner
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset
from training.mixture.score_factory import ScoreCalculatorFactory

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

GENERATOR_PREFIX = "generator-"
DISCRIMINATOR_PREFIX = "discriminator-"

class LipizzanerClient:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self.cc = ConfigurationContainer.instance()

    def _set_output_dir(self, cc):
        output = cc.output_dir
        dataloader = cc.settings["dataloader"]["dataset_name"]
        job_name = cc.settings["general"]["name"]

        cc.output_dir = os.path.join(output, "distributed", dataloader, job_name, str(os.getpid()))
        os.makedirs(cc.output_dir, exist_ok=True)

    def run(self):
        cc = ConfigurationContainer.instance()
        cc.settings = self.config

        output_base_dir = cc.output_dir
        self._set_output_dir(cc)

        self.cc.settings["general"]["distribution"]["client_id"] = rank
        self.cc.settings["general"]["distribution"]["num_clients"] = size

        # It is not possible to obtain reproducible result for large grid due to nature of asynchronous training
        # But still set seed here to minimize variance
        set_random_seed(
            self.cc.settings["general"]["seed"],
            self.cc.settings["trainer"]["params"]["score"]["cuda"],
        )

        if "logging" in cc.settings["general"] and cc.settings["general"]["logging"]["enabled"]:
            LogHelper.setup(cc.settings["general"]["logging"]["log_level"], cc.output_dir)

        self._logger.info("Distributed training recognized, set log directory to {}".format(cc.output_dir))

        try:
            lipizzaner = Lipizzaner()
            lipizzaner.run(cc.settings["trainer"]["n_iterations"])
        except Exception as ex:
            self._logger.critical("An unhandled error occured while running Lipizzaner: {}".format(ex))
            raise ex
        finally:
            self._gather_results()

    def _gather_results(self):
        self._logger.info("Collecting results from clients...")

        # Initialize node client
        dataloader = self.cc.create_instance(self.cc.settings["dataloader"]["dataset_name"])
        network_factory = self.cc.create_instance(
            self.cc.settings["network"]["name"],
            dataloader.n_input_neurons,
            num_classes=dataloader.num_classes,
        )

        # Add gather MPI call here

        if rank != 0:
            return

        scores = []
        for (
            node,
            generator_pop,
            discriminator_pop,
            weights_generator,
            weights_discriminator,
        ) in results:
            node_name = "{}:{}".format(node["address"], node["port"])
            try:
                output_dir = self.get_and_create_output_dir(node)

                for generator in generator_pop.individuals:
                    filename = generator.save_genome(
                        GENERATOR_PREFIX,
                        output_dir,
                    )
                    with open(os.path.join(output_dir, "mixture.yml"), "a") as file:
                        file.write("{}: {}\n".format(filename, weights_generator[generator.source]))

                for discriminator in discriminator_pop.individuals:
                    discriminator.save_genome(
                        DISCRIMINATOR_PREFIX, output_dir, "SemiSupervised" in discriminator.genome.name
                    )

                # Save images
                dataset = MixedGeneratorDataset(
                    generator_pop,
                    weights_generator,
                    self.cc.settings["master"]["score_sample_size"],
                    self.cc.settings["trainer"]["mixture_generator_samples_mode"],
                )

                if "gaussian_" in self.cc.settings["dataloader"]["dataset_name"]:  # Gaussian 2D
                    image_paths = self.save_samples(dataset, output_dir, dataloader, 100000, 10000)
                elif "gaussian" == self.cc.settings["dataloader"]["dataset_name"]:  # Gaussian 1D
                    image_paths = self.save_samples(dataset, output_dir, dataloader)
                else:
                    image_paths = self.save_samples(dataset, output_dir, dataloader)
                self._logger.info(
                    "Saved mixture result images of client {} to target directory {}.".format(node_name, output_dir)
                )

                # Calculate inception or FID score
                score = float("-inf")
                if self.cc.settings["master"]["calculate_score"]:
                    calc = ScoreCalculatorFactory.create()
                    self._logger.info("Score calculator: {}".format(type(calc).__name__))
                    self._logger.info(
                        "Calculating score score of {}. Depending on the type, this may take very long.".format(
                            node_name
                        )
                    )

                    score = calc.calculate(dataset)
                    self._logger.info(
                        "Node {} with weights {} yielded a score of {}".format(node_name, weights_generator, score)
                    )
                    scores.append((node, score))

            except Exception as ex:
                self._logger.error("An error occured while trying to gather results from {}: {}".format(node_name, ex))
                traceback.print_exc()

        if self.cc.settings["master"]["calculate_score"] and scores:
            best_node = sorted(
                scores,
                key=lambda x: x[1],
                reverse=ScoreCalculatorFactory.create().is_reversed,
            )[-1]
            self._logger.info(
                "Best result: {}:{} = {}".format(best_node[0]["address"], best_node[0]["port"], best_node[1])
            )

    def get_and_create_output_dir(self, node):
        directory = os.path.join(
            self.cc.output_dir,
            "master",
            self.cc.settings["general"]["distribution"]["start_time"],
            "{}-{}".format(node["address"], node["port"]),
        )
        os.makedirs(directory, exist_ok=True)
        return directory

    def save_samples(
        self,
        dataset,
        output_dir,
        image_specific_loader,
        n_images=10,
        batch_size=100,
    ):
        image_format = self.cc.settings["general"]["logging"]["image_format"]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        loaded = image_specific_loader.load()
        paths = []

        for i, data in enumerate(dataloader):
            shape = loaded.dataset.train_data.shape if hasattr(loaded.dataset, "train_data") else None
            path = os.path.join(output_dir, "mixture-{}.{}".format(i + 1, image_format))
            image_specific_loader.save_images(Variable(data), shape, path)
            paths.append(path)

            if i + 1 == n_images:
                break

        return paths

