import logging

import os
import numpy as np
import torch
import torch.utils.data

from distribution.client_environment import ClientEnvironment
from helpers.reproducible_helpers import set_random_seed, get_heuristic_seed
from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import is_cuda_available
from helpers.network_helpers import local_private_ip


class Lipizzaner:
    """
    Lipizzaner is a toolkit that trains generative adversarial networks with coevolutionary methods.

    For more details about its usage, visit the GitHub page:
    https://github.com/ALFA-group/lipizzaner-gan
    """

    _logger = logging.getLogger(__name__)

    def __init__(self, trainer=None):
        """
        :param trainer: An implementation of NeuralNetworkTrainer that will be used to train both networks.
        Read from config if None.
        """
        if trainer is not None:
            self.trainer = trainer
        else:
            self.cc = ConfigurationContainer.instance()
            dataloader = self.cc.create_instance(self.cc.settings['dataloader']['dataset_name'])
            network_factory = self.cc.create_instance(self.cc.settings['network']['name'], dataloader.n_input_neurons)
            self.trainer = self.cc.create_instance(self.cc.settings['trainer']['name'], dataloader, network_factory)

        if 'params' in self.cc.settings['trainer'] and 'score' in self.cc.settings['trainer']['params']:
            self.cuda = self.cc.settings['trainer']['params']['score']['cuda']
        else:
            self.cuda = False

        self._logger.info("Parameters: {}".format(self.cc.settings))

        if is_cuda_available() and self.cuda:
            self._logger.info("CUDA is supported on this device and will be used.")
        elif is_cuda_available() and (not self.cuda):
            self._logger.info("CUDA is supported on this device but will NOT be used.")
        else:
            self._logger.info("CUDA is not supported on this device.")

        # It is not possible to obtain reproducible result for large grid due to nature of asynchronous training
        # But still set seed here to minimize variance
        final_seed = get_heuristic_seed(self.cc.settings['general']['seed'], local_private_ip(), ClientEnvironment.port)
        set_random_seed(final_seed, self.cuda)
        self._logger.info("Seed used: {}".format(final_seed))

    def run(self, n_iterations, stop_event=None):
        self._logger.info("Starting training for {} iterations/epochs.".format(n_iterations))
        (generator, g_loss), (discriminator, d_loss) = self.trainer.train(n_iterations, stop_event)
        self._logger.info("Finished training process, f(d)={}, f(g)={}".format(float(d_loss),float(g_loss)))

        # Save the trained parameters
        torch.save(generator.net.state_dict(), os.path.join(self.cc.output_dir, 'generator.pkl'))
        torch.save(discriminator.net.state_dict(), os.path.join(self.cc.output_dir, 'discriminator.pkl'))

