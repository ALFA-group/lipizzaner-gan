import os
import base64
import copy
import logging
import math

from bson import ObjectId, Binary
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from helpers.configuration_container import ConfigurationContainer
from helpers.network_helpers import local_ip_address

CONNECTION_TIMEOUT = 3000


class DbLogger:
    """
    Class write log entries into a MongoDB database, which later can e.g. be used for analytics
    """

    _logger = logging.getLogger(__name__)

    def __init__(self, server_address=None, current_experiment=None):
        self.current_experiment = ObjectId(current_experiment)
        self.settings = ConfigurationContainer.instance().settings

        if server_address is None:
            server_address = self.settings['general']['logging'].get('log_server', None)
            if server_address is None:
                self.is_enabled = False
                return

        self.database = MongoClient(server_address, serverSelectionTimeoutMS=CONNECTION_TIMEOUT).lipizzaner_db
        self.is_enabled = True

    def create_experiment(self, settings):
        """
        Creates a new experiment database entry. Currently only implemented for square grids.
        :return: The database ID of the new entry
        """
        name = settings['general']['distribution']['start_time']
        master = local_ip_address()
        grid_size = len(settings['general']['distribution']['client_nodes'])
        dim = round(math.sqrt(grid_size))

        # Don't log connection string of log server
        settings = copy.deepcopy(settings)
        settings['general']['logging'].pop('log_server', None)

        experiment = {
            'name': name,
            'master': master,
            'topology': {
                'type': 'grid',
                'width': dim,
                'height': dim,
            },
            'settings': {key: settings[key] for key in settings if key != 'logging'}
        }

        collection = self.database.experiments
        return str(collection.insert_one(experiment).inserted_id)

    def finish_experiment(self, experiment_id):
        collection = self.database.experiments
        collection.update_one({'_id': ObjectId(experiment_id)}, {'$set': {'end_time': datetime.now(timezone.utc)}},
                              upsert=False)

    def add_experiment_results(self, experiment_id, node_name, image_paths, inception_score):
        collection = self.database.experiments
        images = [self._load_images(path) for path in image_paths]
        collection.update_one({'_id': ObjectId(experiment_id)},
                              {
                                  '$push': {
                                      'results': {
                                          'images': images,
                                          'inception_score': inception_score,
                                          'mixture_center': node_name
                                      }
                                  }
                              },
                              upsert=False)

    def log_results(self, iteration, neighbourhood, concurrent_populations, inception_score, duration_sec,
                    path_real_images, path_fake_images):
        """
        Adds a new log database entry with the given state, linked to the current experiment
        """
        if self.current_experiment is None:
            raise Exception('Cannot log to database, as no active experiment is set.')

        if os.path.exists(path_fake_images):
            fake_images = self._load_images(path_fake_images)
        else:
            fake_images = None

        log_entry = {
            'experiment_id': self.current_experiment,
            'iteration': iteration,
            'grid_position': {
                'x': neighbourhood.grid_position[0],
                'y': neighbourhood.grid_position[1],
            },
            'node_name': neighbourhood.local_node['id'],
            'mixture_weights_gen': list(neighbourhood.mixture_weights_generators.values()),
            'mixture_weights_dis': list(neighbourhood.mixture_weights_discriminators.values()),
            'inception_score': inception_score,
            'duration_sec': duration_sec,
            'generators': [self._parse_individual(g) for g in concurrent_populations.generator.individuals],
            'discriminators': [self._parse_individual(d) for d in concurrent_populations.discriminator.individuals],
            'fake_images': fake_images
        }

        if iteration == 0:
            if os.path.exists(path_real_images):
                real_images = self._load_images(path_real_images)
            else:
                real_images = None
                
            log_entry['real_images'] = real_images

        try:
            collection = self.database.log_entries
            collection.insert_one(log_entry)
        except ServerSelectionTimeoutError:
            self._logger.error('Could not write log entry to database server, as it was not reachable in {}ms.'.format(
                CONNECTION_TIMEOUT
            ))

    @staticmethod
    def _parse_individual(individual):
        return {
            'cell_id': individual.id,
            'loss': individual.fitness,
            'hyper_params': {'lr': individual.learning_rate}
        }

    @staticmethod
    def _load_images(path_fake_images):
        with open(path_fake_images, 'rb') as fin:
            return base64.b64encode(fin.read()).decode('utf-8')
