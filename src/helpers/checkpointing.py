from os.path import dirname
import glob
import yaml
import os
import logging


#path = '/media/toutouh/224001034000DF81/lipi-gan-public-checkpointing/lipizzaner-gan/src/output/lipizzaner_gan/distributed/mnist/2019-11-22_11-49-00/'


def create_cell_info(source):
    splitted_source = source.split(':')
    return {'address': splitted_source[0], 'port': splitted_source[1], 'id': source}

class ExperimentResuming():


    _logger = logging.getLogger(__name__)

    def __init__(self, experiment_path):
        self.checkpoints_storage = self.create_checkpoints_storage(experiment_path)

    def create_checkpoints_storage(self, experiment_path):
        assert os.path.isdir(experiment_path), 'Checkpoint of experiment in folder {} not found.'.format(experiment_path)
        self._logger.info('Recovering checkpoint information from checkpoint of the experiment stored in {}'.format(experiment_path))
        return [self.get_checkpoint_structure(checkpoint_file) for checkpoint_file in glob.glob(experiment_path + '*/checkpoint*.yml')]

    def get_checkpoint_structure(self, checkpoint_file):
        assert os.path.isfile(checkpoint_file)

        def get_local_individuals(individual_type, checkpoint_data):
             return [individual['source'] for individual in checkpoint_data[individual_type]['individuals'] if individual['is_local']]

        def get_adjacent_individuals(individual_type, checkpoint_data):
              return [create_cell_info(individual['source']) for individual in checkpoint_data[individual_type]['individuals'] if
                    not individual['is_local']]

        def get_learning_rate(individual_type, checkpoint_data):
             return checkpoint_data[individual_type]['learning_rate']

        checkpoint_data = yaml.load(open(checkpoint_file))
        checkpoint = dict()
        checkpoint['local_generators'] = get_local_individuals('generators', checkpoint_data)
        checkpoint['local_discriminators'] = get_local_individuals('discriminators', checkpoint_data)
        checkpoint['discriminators_learning_rate'] = get_learning_rate('discriminators', checkpoint_data)
        checkpoint['generators_learning_rate'] = get_learning_rate('generators', checkpoint_data)
        checkpoint['adjacent_individuals'] = get_adjacent_individuals('generators', checkpoint_data)

        checkpoint['iteration'] = checkpoint_data['iteration']
        checkpoint['time'] = checkpoint_data['time']
        checkpoint['id'] = checkpoint_data['id']
        checkpoint['path'] = dirname(checkpoint_file)
        checkpoint['grid_size'] = checkpoint_data['grid_size']
        checkpoint['position'] = (checkpoint_data['position']['x'], checkpoint_data['position']['y'])

        self._logger.info(
            'Recovered checkpoint information from checkpoint file {}.\nCheckpoint information: {}'.format(checkpoint_file, checkpoint))

        return checkpoint

    def get_population_cell_info(self):
        client_soueces = self.get_population_sources()
        clients = []
        for source in client_soueces:
            clients.append(create_cell_info(source))
        return clients

    def get_population_sources(self):
        sources = set()
        for checkpoint in self.checkpoints_storage:
            [sources.add(local_genarator) for local_genarator in checkpoint['local_generators']]
            [sources.add(local_discriminator) for local_discriminator in checkpoint['local_discriminators']]
        return list(sources)

    def get_local_generators_paths(self, source):
        assert source in self.get_population_sources()
        for checkpoint in self.checkpoints_storage:
            for local_source in checkpoint['local_generators']:
                if source == local_source:
                    return glob.glob(checkpoint['path'] + '/generator-*.pkl')

    def get_local_discriminators_paths(self, source):
        assert source in self.get_population_sources()
        for checkpoint in self.checkpoints_storage:
            for local_source in checkpoint['local_discriminators']:
                if source == local_source:
                    return glob.glob(checkpoint['path'] + '/discriminator-*.pkl')

    def get_iterations(self, source):
        assert source in self.get_population_sources()
        for checkpoint in self.checkpoints_storage:
            for local_source in checkpoint['local_discriminators']:
                if source == local_source:
                    return checkpoint['iteration']

    def get_id(self, source):
        assert source in self.get_population_sources()
        for checkpoint in self.checkpoints_storage:
            for local_source in checkpoint['local_discriminators']:
                if source == local_source:
                    return checkpoint['id']

    def get_iterations_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return [checkpoint['iteration'] for checkpoint in self.checkpoints_storage if checkpoint['id'] == id][0]

    def get_local_generators_paths_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return [glob.glob(checkpoint['path'] + '/generator-*.pkl') for checkpoint in self.checkpoints_storage if
                checkpoint['id'] == id][0]

    def get_local_discriminators_paths_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return [glob.glob(checkpoint['path'] + '/discriminator-*.pkl') for checkpoint in self.checkpoints_storage if
                checkpoint['id'] == id][0]

    def get_discriminators_learning_rate_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return [checkpoint['discriminators_learning_rate'] for checkpoint in self.checkpoints_storage if checkpoint['id'] == id][0]

    def get_generators_learning_rate_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return [checkpoint['generators_learning_rate'] for checkpoint in self.checkpoints_storage if checkpoint['id'] == id][0]

    def get_adjacent_cells_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return \
        [checkpoint['adjacent_individuals'] for checkpoint in self.checkpoints_storage if checkpoint['id'] == id][0]

    def get_topology_details_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return \
            [{'grid_size': checkpoint['grid_size'], 'position': checkpoint['position'], 'cell_info': create_cell_info(checkpoint['local_generators'][0])}  for checkpoint in self.checkpoints_storage if checkpoint['id'] == id][0]

    def get_checkpoint_data_id(self, id):
        assert 0 <= id <= len(self.checkpoints_storage)
        return {'iteration': self.get_iterations_id(id),
                'generators_path': self.get_local_generators_paths_id(id),
                'discriminators_path': self.get_local_discriminators_paths_id(id),
                'generators_learning_rate': self.get_generators_learning_rate_id(id),
                'discriminators_learning_rate': self.get_discriminators_learning_rate_id(id),
                'adjacent_cells': self.get_adjacent_cells_id(id),
                'topology_details': self.get_topology_details_id(id),
                'cell_number': id}

