import logging
import math
import os
import sys

import yaml
from mpi4py import MPI

from helpers.configuration_container import ConfigurationContainer
from helpers.log_helper import LogHelper
from helpers.math_helpers import is_square
from helpers.yaml_include_loader import YamlIncludeEnvLoader
from lipizzaner_client_mpi import LipizzanerClient
from lipizzaner_master_mpi import LipizzanerMaster

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

logger = logging.getLogger(__name__)


def read_settings(config_filepath):
    with open(config_filepath, "r") as config_file:
        return yaml.load(config_file, YamlIncludeEnvLoader)

def initialize_settings():
    cc = ConfigurationContainer.instance()
    cc.settings = read_settings(sys.argv[1])
    if "logging" in cc.settings["general"] and cc.settings["general"]["logging"]["enabled"]:
        log_dir = os.path.join(cc.settings["general"]["output_dir"], "log")
        LogHelper.setup(cc.settings["general"]["logging"]["log_level"], log_dir)
    return cc

if __name__ == "__main__":
    initialize_settings()
    
    if size == 0 or not is_square(size):
        msg = "{} clients found, but Lipizzaner currently only supports square grids.".format(size)
        logger.critical(msg)
        raise Exception(msg)

    LipizzanerClient().run()
