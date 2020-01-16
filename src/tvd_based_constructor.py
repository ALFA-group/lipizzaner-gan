import numpy as np
from itertools import product
import glob
import random

class TVDBasedConstructor:
    def __init__(self, precision=10, generators_path='./generators/', mode='iterative'):
        self.list_of_weights = []
        self.list_of_generators = []
        # self.current_outputs_probability = len(self.real_probability_distribution) * [0]
        self.precision = precision
        self.possible_weights = self.create_weights_list(0, self.precision+1, 1)

        self.generator_index = 0
        self.generators_path = generators_path
        self.generators_in_path = self.get_all_generators_in_path()
        self.mode = mode

    def get_all_generators_in_path(self):
        return len([gen for gen in glob.glob('{}*gene*.pkl'.format(self.generators_path))])

    def create_weights_list(self, min_value, max_value, step_size):
        return np.arange(min_value, max_value, step_size)

    def get_next_generator_path(self):
        if self.mode=='greddyiterative':
            path = '{}mnist-generator-{:03d}.pkl'.format(self.generators_path, self.generator_index)
            source = '{:03d}'.format(self.generator_index)
            self.generator_index += 1
        elif self.mode=='random':
            generator_index = random.randint(0, self.generators_in_path)
            path = '{}mnist-generator-{:03d}.pkl'.format(self.generators_path, generator_index)
            source = '{:03d}'.format(generator_index)
        return path, source


    def get_weights_tentative(self, size):
        tentative_weights = list(product(list(self.possible_weights), repeat=size))
        result = []
        for w in tentative_weights:
            if sum(w)==self.precision:
                result.append(list(w))
        return np.array(result)/self.precision, len(result)

    def get_next_generator(self, criterion):
        return None

    # def create_ensamble(self, ensemble_size, criterion, step_size):
    #     self.list_of_weights = [1.0]
    #     self.list_of_generators = self.getNextGenerator(criterion)
