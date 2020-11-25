import json
import logging
import os
import re 
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
    _last_checkpoint = "" # initialize to empty timestamp 
    _lipizzaner = None 

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
        
        cc = ConfigurationContainer.instance()
        path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)

        if ClientAPI.is_busy:
            if os.path.exists(path):
                ClientAPI._logger.info('Experiments DEL: sleep file found for client ' + str(ClientEnvironment.port))
                response = Response() 
            else :
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

        cc = ConfigurationContainer.instance()
        path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)

        if ClientAPI.is_busy:
            if os.path.exists(path):
                ClientAPI._logger.info('Experiments GET: sleep file found for client ' + str(ClientEnvironment.port))
                response = Response() 
            else :
                ClientAPI._logger.info('Sending neighbourhood results to master')
                response = jsonify(ClientAPI._gather_results())
                ClientAPI._finish_event.set()

        else:
            ClientAPI._logger.warning('Master requested results, but no experiment is running.')
            response = Response()

        ClientAPI._lock.release()

        return response

    @staticmethod
    @app.route('/experiments/sleep', methods=['GET'])
    # used to create sleepfile if one doesn't exist and delete it if one does (toggle)
    def sleep(): 
        ClientAPI._lock.acquire()

        response = Response()

        cc = ConfigurationContainer.instance()
        if hasattr(cc, "settings"):
            output_base_dir = cc.output_dir 
            path = output_base_dir + "/sleepfile.txt" + str(ClientEnvironment.port)
            # TODO set flag in client environment to be accessed in lipi gan trainer 
        
        ClientAPI._logger.info("Sleep Request made with sleepfile at {}".format(path))

        if os.path.exists(path):
            ClientAPI._logger.info("Removing sleep file or waking up client")
            os.remove(path)
        else:
            ClientAPI._logger.info("Creating sleep file or killing client")
            with open(path, 'w+') as f:
                f.write("sleep")
        ClientAPI._lock.release()
        return response

    @staticmethod
    @app.route('/replaceNeighbor', methods=['POST'])
    def replace_neighbor():
        config = request.get_json()
        ClientAPI._lock.acquire()

        response = Response()

        cc = ConfigurationContainer.instance()
        cc.settings = config
        path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)

        if ClientAPI.is_busy:
            if os.path.exists(path):
                ClientAPI._logger.info('Replace Neighbor: sleep file found for client ' + str(ClientEnvironment.port))
                return Response() 
        
        dead_port = cc.settings['general']['distribution']['dead_port']
        replacement_port = cc.settings['general']['distribution']['replacement']

        ClientAPI._lipizzaner.replace_neighbor(dead_port, replacement_port)

        return response


    # NEW METHOD for requesting checkpoint from each cell 
    @staticmethod
    @app.route('/experiments/checkpoint', methods=['GET']) 
    def get_checkpoints(): 
        ClientAPI._lock.acquire()

        if ClientAPI.is_busy:
            # get the checkpoint
            ClientAPI._logger.info('Sending checkpoint to master')
            response = Response() 
            checkpoint = ClientAPI._lipizzaner.trainer._checkpoint # TODO: does this work for getting dict
            result = {
                'checkpoint': checkpoint
            }
            return jsonify(result)
        else: 
            ClientAPI._logger.warning('Master requested checkpoints, but no experiment is running.')
            response = Response()
        
        ClientAPI._lock.release()
        return response

    @staticmethod
    @app.route('/status', methods=['GET'])
    def get_status():
        # makes Master register client as dead if sleepfile exists
        #TODO
        # take timestamp of latest checkpoint to see if we need to
        # request checkpoint -> look into output dir for the 
        # new checkpoint 
        cc = ConfigurationContainer.instance()

        # ClientAPI._logger.info('CC dictionary is {}'.format(cc.__dict__))
        result = {
            'busy': ClientAPI.is_busy,
            'finished': ClientAPI.is_finished
        }

        if hasattr(cc, "settings"):
            if 'general' in cc.settings.keys():
                sleep_path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)
                if os.path.exists(sleep_path) :
                        ClientAPI._logger.info('STATUS Client made to sleep ' + str(ClientEnvironment.port))
                        response = Response()
                        response._status_code = 404 
                        return response
        #         # otherwise find files in output dir and sort in alphanumeric order
        #         # return timestamp of most recent checkpoint 
        #         files = [x for x in os.listdir(cc.output_dir) if x.startswith("checkpoint")]
        #         sortedFiles = sorted_nicely(files)
        #         if len(sortedFiles) > 0:
        #             latestCheckpoint = sortedFiles[len(sortedFiles) - 1] # last one should be latest
        #             if ClientAPI._last_checkpoint != latestCheckpoint:
        #                 # TODO: not sure whether to update the last checkpoint time here so i can avoid herd effect
        #                 # but i also don't want to miss checkpoints if the call fails 
        #                 result['new_checkpoint'] = True 
                
        return jsonify(result)

    @staticmethod
    @app.route('/parameters/discriminators', methods=['GET'])
    def get_discriminators():
        populations = ConcurrentPopulations.instance()

        cc = ConfigurationContainer.instance()
        path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)

        if ClientAPI.is_busy: # not sure if I should have this check
            if os.path.exists(path):
                ClientAPI._logger.info('Discriminators GET: sleep file found for client ' + str(ClientEnvironment.port))
                return Response() 

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
        
        cc = ConfigurationContainer.instance()
        path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)

        if ClientAPI.is_busy:
            if os.path.exists(path):
                ClientAPI._logger.info('Discriminators BEST GET: sleep file found for client ' + str(ClientEnvironment.port))
                return Response() 
            
        ClientAPI._logger.info('Discriminators BEST GET for ' + str(ClientEnvironment.port))
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

        cc = ConfigurationContainer.instance()
        path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)

        if ClientAPI.is_busy:
            if os.path.exists(path):
                ClientAPI._logger.info('Generators GET: sleep file found for client ' + str(ClientEnvironment.port))
                return Response()

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

        cc = ConfigurationContainer.instance()
        path = cc.output_dir + "/sleepfile.txt" + str(ClientEnvironment.port)

        if ClientAPI.is_busy:
            if os.path.exists(path):
                ClientAPI._logger.info('Generators BEST GET: sleep file found for client ' + str(ClientEnvironment.port))
                return Response()

        if populations.generator is not None:
            best_individual = sorted(populations.generator.individuals, key=lambda x: x.fitness)[0]
            parameters = [ClientAPI._individual_to_json(best_individual)]
        else:
            parameters = []

        data = json.dumps(parameters)
        return Response(response=data, status=200, mimetype="application/json")

    @staticmethod
    def _run_lipizzaner(config, iterations=None):
        LogHelper.log_only_flask_warnings()

        cc = ConfigurationContainer.instance()
        cc.settings = config

        output_base_dir = cc.output_dir
        ClientAPI._set_output_dir(cc)

        if 'logging' in cc.settings['general'] and cc.settings['general']['logging']['enabled']:
            LogHelper.setup(cc.settings['general']['logging']['log_level'], cc.output_dir)

        ClientAPI._logger.info('Distributed training recognized, set log directory to {}'.format(cc.output_dir))

        try:
            if 'neighbors' in cc.settings['general']['distribution']:
                lipizzaner = Lipizzaner(_neighbors=cc.settings['general']['distribution']['neighbors'])
            else:
                lipizzaner = Lipizzaner()
    
            ClientAPI._lipizzaner = lipizzaner # NEW saving the instance here to access checkpoint later 
            # initialize lipizzaner_gan_trainer instance and neighborhood 
            n_iterations = cc.settings['trainer']['n_iterations']
            neighbors = None
            if iterations != None:
                n_iterations = iterations 
                neighbors = cc.settings['general']['distribution']['neighbors'] 
            lipizzaner.run(n_iterations, ClientAPI._stop_event)
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
            cc.output_dir = output_base_dir
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
        neighbourhood = Neighbourhood() # Neighbourhood.instance()
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

    # NEW METHOD for gathering checkpoints
    @staticmethod
    def _get_checkpoints():
        neighborhood = Neighbourhood.instance()
        cc = ConfigurationContainer.instance()
        checkpoints = {} # TODO
        return checkpoints

    @classmethod
    def _set_output_dir(cls, cc):
        output = cc.output_dir
        dataloader = cc.settings['dataloader']['dataset_name']
        start_time = cc.settings['general']['distribution']['start_time']

        cc.output_dir = os.path.join(output, 'distributed', dataloader, start_time, str(os.getpid()))
        os.makedirs(cc.output_dir, exist_ok=True)

    def listen(self, port):
        ClientEnvironment.port = port
        ConcurrentPopulations.instance().lock()
        self.app.run(threaded=True, port=port, host="0.0.0.0")

# sorts alphanumerically so '4 sheets', '12 sheets', 'booklet' would be the order with numbers 
# first in increasing order 
def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)