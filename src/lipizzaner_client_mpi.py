
import logging
from threading import Event, Lock, Thread

from distribution.concurrent_populations import ConcurrentPopulations
from helpers.configuration_container import ConfigurationContainer
from helpers.log_helper import LogHelper
from helpers.or_event import or_event
from lipizzaner import Lipizzaner


class LipizzanerClient:
    def __init__(self, config):
        self.config = config

        self._logger = logging.getLogger(__name__)

    def run(self):
        cc = ConfigurationContainer.instance()
        cc.settings = self.config

        output_base_dir = cc.output_dir
        self._set_output_dir(cc)

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
            self._logger.info("Finished experiment, waiting for new requests.")
            cc.output_dir = output_base_dir
            ConcurrentPopulations.instance().lock()
