import json
import logging
import os
import traceback
from threading import Thread, Lock, Event

from flask import Flask, request, Response, jsonify

from distribution.client_environment import ClientEnvironment
from distribution.concurrent_populations import ConcurrentPopulations
from distribution.neighbourhood import Neighbourhood
from distribution.state_encoder import StateEncoder
from helpers.configuration_container import ConfigurationContainer
from helpers.log_helper import LogHelper
from helpers.or_event import or_event
from lipizzaner import Lipizzaner


class ClientAPI:
    app = Flask(__name__)
    is_busy = False
    is_finished = False

    _stop_event = None
    _finish_event = None
    _lock = Lock()

    _logger = logging.getLogger(__name__)

    @staticmethod
    @app.route('/experiments', methods=['POST'])
    def run_experiment():
        config = request.get_json()
        ClientAPI._lock.acquire()
        if ClientAPI.is_busy:
            ClientAPI._lock.release()
            return 'Client is currently busy.', 500

        ClientAPI.is_finished = False
        ClientAPI.is_busy = True
        ClientAPI._lock.release()
        ClientAPI._stop_event = Event()
        ClientAPI._finish_event = Event()
        worker_thread = Thread(target=ClientAPI._run_lipizzaner, args=(config,))
        worker_thread.start()

        return Response()

    @staticmethod
    @app.route('/experiments', methods=['DELETE'])
    def terminate_experiment():
        ClientAPI._lock.acquire()

        if ClientAPI.is_busy:
            ClientAPI._logger.warning('Received stop signal from master, experiment will be quit.')
            ClientAPI._stop_event.set()
        else:
            ClientAPI._logger.warning('Received stop signal from master, but no experiment is running.')

        ClientAPI._lock.release()
        return Response()

    @staticmethod
    @app.route('/experiments', methods=['GET'])
    def get_results():
        ClientAPI._lock.acquire()

        if ClientAPI.is_busy or ClientAPI.is_finished:
            ClientAPI._logger.info('Sending neighbourhood results to master')
            response = jsonify(ClientAPI._gather_results())
            #ClientAPI._finish_event.set()
        else:
            ClientAPI._logger.warning('Master requested results, but no experiment is running.')
            response = Response()

        ClientAPI._lock.release()

        return response

    @staticmethod
    @app.route('/scores', methods=['GET'])
    def get_scores():
        ClientAPI._lock.acquire()

        if ClientAPI.is_finished:
            ClientAPI._logger.info('Sending client score to master')
            response = jsonify(ClientAPI._gather_scores())
            ClientAPI._finish_event.set()
        else:
            ClientAPI._logger.warning('Master requested scores, but experiemtn not finished.')
            response = Response()

        ClientAPI._lock.release()

        return response

    @staticmethod
    @app.route('/status', methods=['GET'])
    def get_status():
        result = {
            'busy': ClientAPI.is_busy,
            'finished': ClientAPI.is_finished
        }
        return jsonify(result)

    @staticmethod
    @app.route('/parameters/discriminators', methods=['GET'])
    def get_discriminators():
        populations = ConcurrentPopulations.instance()

        populations.lock()
        if populations.discriminator is not None:
            parameters = [ClientAPI._individual_to_json(i) for i in populations.discriminator.individuals]
        else:
            parameters = []
        populations.unlock()

        data = json.dumps(parameters)
        return Response(response=data, status=200, mimetype="application/json")

    @staticmethod
    @app.route('/parameters/discriminators/best', methods=['GET'])
    def get_best_discriminator():
        populations = ConcurrentPopulations.instance()

        populations.lock()
        if populations.discriminator is not None:
            best_individual = sorted(populations.discriminator.individuals, key=lambda x: x.fitness)[0]
            parameters = [ClientAPI._individual_to_json(best_individual)]
        else:
            parameters = []
        populations.unlock()

        data = json.dumps(parameters)
        return Response(response=data, status=200, mimetype="application/json")

    @staticmethod
    @app.route('/parameters/generators', methods=['GET'])
    def get_generators():
        populations = ConcurrentPopulations.instance()

        populations.lock()
        if populations.generator is not None:
            parameters = [ClientAPI._individual_to_json(i) for i in populations.generator.individuals]
        else:
            parameters = []
        populations.unlock()

        data = json.dumps(parameters)
        return Response(response=data, status=200, mimetype="application/json")

    @staticmethod
    @app.route('/parameters/generators/best', methods=['GET'])
    def get_best_generator():
        populations = ConcurrentPopulations.instance()

        if populations.generator is not None:
            best_individual = sorted(populations.generator.individuals, key=lambda x: x.fitness)[0]
            parameters = [ClientAPI._individual_to_json(best_individual)]
        else:
            parameters = []

        data = json.dumps(parameters)
        return Response(response=data, status=200, mimetype="application/json")

    @staticmethod
    def _run_lipizzaner(config):
        LogHelper.log_only_flask_warnings()

        cc = ConfigurationContainer.instance()
        cc.settings = config

        output_dir = ClientAPI._get_output_dir(cc)

        if 'logging' in cc.settings['general'] and cc.settings['general']['logging']['enabled']:
            LogHelper.setup(cc.settings['general']['logging']['log_level'], output_dir)

        ClientAPI._logger.info('Distributed training recognized, set log directory to {}'.format(output_dir))

        try:
            lipizzaner = Lipizzaner()
            lipizzaner.run(cc.settings['trainer']['n_iterations'], ClientAPI._stop_event)
            ClientAPI.is_finished = True

            # Wait until master finishes experiment, i.e. collects results, or experiment is terminated
            or_event(ClientAPI._finish_event, ClientAPI._stop_event).wait()
        except Exception as ex:
            ClientAPI.is_finished = True
            ClientAPI._logger.critical('An unhandled error occured while running Lipizzaner: {}'.format(ex))
            # Flask 1.0.2 does not print the stack trace of exceptions anymore
            traceback.print_exc()
            raise ex
        finally:
            ClientAPI.is_busy = False
            ClientAPI._logger.info('Finished experiment, waiting for new requests.')
            ConcurrentPopulations.instance().lock()

    @staticmethod
    def _individual_to_json(individual):
        json_response = {
            'id': individual.id,
            'parameters': individual.genome.encoded_parameters,
            'learning_rate': individual.learning_rate,
            'optimizer_state': StateEncoder.encode(individual.optimizer_state)
        }
        if individual.iteration is not None:
            json_response['iteration'] = individual.iteration

        return json_response

    @staticmethod
    def _gather_results():
        neighbourhood = Neighbourhood.instance()
        cc = ConfigurationContainer.instance()
        results = {
            'generators': neighbourhood.best_generator_parameters,
            'discriminators': neighbourhood.best_discriminator_parameters,
            'weights_generators': neighbourhood.mixture_weights_generators
        }
        if cc.settings['trainer']['name'] == 'with_disc_mixture_wgan' \
            or cc.settings['trainer']['name'] == 'with_disc_mixture_gan':
            results['weights_discriminators'] = neighbourhood.mixture_weights_discriminators
        else:
            results['weights_discriminators'] = 0.0

        return results

    @staticmethod
    def _gather_scores():
        neighbourhood = Neighbourhood.instance()
        cc = ConfigurationContainer.instance()
        results = {
            'score': neighbourhood.score
        }

        return results

    @classmethod
    def _get_output_dir(cls, cc):
        output = cc.output_dir
        dataloader = cc.settings['dataloader']['dataset_name']
        start_time = cc.settings['general']['distribution']['start_time']

        output_dir = os.path.join(output, 'distributed', dataloader, start_time, str(os.getpid()))

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def listen(self, port):
        ClientEnvironment.port = port
        ConcurrentPopulations.instance().lock()
        self.app.run(threaded=True, port=port, host="0.0.0.0")
