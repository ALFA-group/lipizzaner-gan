import numpy as np
from itertools import product
import glob
import random

class TVDBasedRandomSearch:
    def __init__(self, precision=10, generators_path='./generators/', mode='iterative'):
        self.precision = precision
        self.possible_weights = self.create_weights_list(0, self.precision+1, 1)
        self.generators_path = generators_path
        self.generators_in_path = self.get_all_generators_in_path()

    def get_all_generators_in_path(self):
        return len([gen for gen in glob.glob('{}*gene*.pkl'.format(self.generators_path))])

    def create_weights_list(self, min_value, max_value, step_size):
        return np.arange(min_value, max_value, step_size)

    def get_genarators_for_ensemble(self, size):
        generator_indizes = random.sample(range(0, self.generators_in_path), size)
        return ['{}mnist-generator-{:03d}.pkl'.format(self.generators_path, generator_index) for generator_index in generator_indizes], \
               ['{:03d}'.format(generator_index) for generator_index in generator_indizes]

    def get_possible_tentative_weights(self, current_prob, i, size):
        max_prob = 1 * self.precision
        number_of_following_weights = size - i
        if number_of_following_weights==1:
            return [max_prob - current_prob]
        else:
            max_probability = (max_prob - current_prob) / number_of_following_weights
            return self.possible_weights[self.possible_weights <= max_probability]

    def get_weights_tentative(self, size):
        result = []
        for i in range(size):
            current_prob = sum(result)
            possible_weights = self.get_possible_tentative_weights(current_prob, i, size)
            result.append(random.choice(possible_weights))
            i += 1
        return np.array(result) / self.precision, len(result)


