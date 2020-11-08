import logging
import time
from concurrent.futures import as_completed, ThreadPoolExecutor

import requests

from distribution.state_encoder import StateEncoder
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.population import Population, TYPE_GENERATOR, TYPE_DISCRIMINATOR

TIMEOUT_SEC_DEFAULT = 60
MAX_HTTP_CLIENT_THREADS = 5


class NodeClient:
    _logger = logging.getLogger(__name__)

    def __init__(self, network_factory):
        self.network_factory = network_factory
        self.cc = ConfigurationContainer.instance()

    def get_all_generators(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        """
        Concurrently loads all current generator individuals from the given nodes.
        Returns when all are loaded, or raises TimeoutError when timeout is reached.
        """
        generators = self.load_generators_from_api(nodes, timeout_sec)

        return [self._parse_individual(gen, self.network_factory.create_generator)
                for gen in generators if self._is_json_valid(gen)]

    def get_all_discriminators(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        """
        Concurrently loads all current discriminator individuals from the node specified by 'addresses'
        Returns when all are loaded, or raises TimeoutError when timeout is reached.
        """
        discriminators = self.load_discriminators_from_api(nodes, timeout_sec)

        return [self._parse_individual(disc, self.network_factory.create_discriminator)
                for disc in discriminators if self._is_json_valid(disc)]

    def get_best_generators(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        generators = self.load_best_generators_from_api(nodes, timeout_sec)

        return [self._parse_individual(gen, self.network_factory.create_generator)
                for gen in generators if self._is_json_valid(gen)]

    def load_best_generators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, '/parameters/generators/best', timeout_sec)

    def load_generators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, '/parameters/generators', timeout_sec)

    def load_best_discriminators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, '/parameters/discriminators/best', timeout_sec)

    def load_discriminators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, '/parameters/discriminators', timeout_sec)

    @staticmethod
    def _load_results(node, timeout_sec):
        address = 'http://{}:{}/experiments'.format(node['address'], node['port'])
        NodeClient._logger.info('Attempting to load results from {}'.format(address))
        try:
            resp = requests.get(address, timeout=timeout_sec)
            return resp.json()
        except Exception as ex:
            NodeClient._logger.error('Error loading results from {}: {}.'.format(address, ex))
            return None

    def gather_results(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        """
        Gathers the final results from each client node, and therefore finishes their experiment runs.
        :return: A list of result tuples: [(node, generator_population, discriminator_population)]
        """

        results = []
        with ThreadPoolExecutor(max_workers=MAX_HTTP_CLIENT_THREADS) as executor:

            futures = {executor.submit(self._load_results, node, timeout_sec): node for node in nodes}
            for future in as_completed(futures):
                # Result has the form { 'discriminators': [[],  [], ..], 'generators': [[], [], ..] }
                node = futures[future]
                result = future.result()
                if result is not None:
                    results.append((node,
                                    self._create_population(result['generators'],
                                                            self.network_factory.create_generator,
                                                            TYPE_GENERATOR),
                                    self._create_population(result['discriminators'],
                                                            self.network_factory.create_discriminator,
                                                            TYPE_DISCRIMINATOR),
                                    result['weights_generators'],
                                    result['weights_discriminators'])),

        return results

    def get_client_statuses(self):
        statuses = []
        for client in self.cc.settings['general']['distribution']['client_nodes']:
            address = 'http://{}:{}/status'.format(client['address'], client['port'])
            try:
                resp = requests.get(address)
                assert resp.status_code == 200
                result = resp.json()
                # TODO if checkpoint newer then call the checkpoint func 
                result['address'] = address
                result['alive'] = True
                result['port'] = client['port']
                statuses.append(result)
                # if there's a new checkpoint from the client then retrieve it 
                # if resp.json()['new_checkpoint']:
                #     checkpoint_addr = 'http://{}:{}/experiments/checkpoint'.format(client['address'], client['port'])
                #     try:
                #         resp_chkpt = requests.get(checkpoint_addr)
                #         assert resp.status_code == 200
                #         # save the checkpoint 
                #         checkpoint_path = self.cc.settings['general']['checkpoint_dir'] + '_{}'.format(client['port']) # todo make it unique for each client and version
                #         with open(checkpoint_path, 'w+') as file:
                #             file.write('{}'.format(resp_chkpt.json()['checkpoint']))
                #         file.close()
                #     except Exception:
                #         pass 
            except Exception:
                statuses.append({
                    'busy': None,
                    'finished': None,
                    'alive': False,
                    'address': client['address'],
                    'port': client['port']
                })

        return statuses

    def stop_running_experiments(self, except_for_clients=None):
        if except_for_clients is None:
            except_for_clients = []

        clients = self.cc.settings['general']['distribution']['client_nodes']
        active_clients = [c for c in clients if not any(d for d in except_for_clients if d['address'] == c['address']
                                                        and d['port'] == c['port'])]
        for client in active_clients:
            address = 'http://{}:{}/experiments'.format(client['address'], client['port'])
            requests.delete(address)

    @staticmethod
    def _load_parameters_async(node, path, timeout_sec):
        address = 'http://{}:{}{}'.format(node['address'], node['port'], path)

        try:
            start = time.time()
            resp = requests.get(address, timeout=timeout_sec).json()
            stop = time.time()
            NodeClient._logger.info('Loading parameters from endpoint {} took {} seconds'.format(address, stop - start))
            for n in resp:
                n['source'] = '{}:{}'.format(node['address'], node['port'])
            return resp
        except Exception as ex:
            NodeClient._logger.error('Error loading parameters from endpoint {}: {}.'.format(address, ex))
            return []

    def _load_parameters_concurrently(self, nodes, path, timeout_sec):
        """
        Returns a list of parameter lists
        """

        all_parameters = []
        with ThreadPoolExecutor(max_workers=MAX_HTTP_CLIENT_THREADS) as executor:
            futures = [executor.submit(self._load_parameters_async, node, path, timeout_sec) for node in nodes]
            for future in as_completed(futures):
                all_parameters.extend(future.result())
        return all_parameters

    @staticmethod
    def _parse_individual(json, create_genome):
        return Individual.decode(create_genome,
                                 json['parameters'],
                                 is_local=False,
                                 learning_rate=json['learning_rate'],
                                 optimizer_state=StateEncoder.decode(json['optimizer_state']),
                                 source=json['source'],
                                 id=json['id'],
                                 iteration=json.get('iteration', None))

    @staticmethod
    def _is_json_valid(json):
        return json and json['parameters'] and len(json['parameters']) > 0

    @staticmethod
    def _create_population(all_parameters, create_genome, population_type):
        individuals = [Individual.decode(create_genome, parameters['parameters'],
                                         source=parameters['source'])
                       for parameters in all_parameters if parameters and len(parameters) > 0]
        return Population(individuals, float('-inf'), population_type)
