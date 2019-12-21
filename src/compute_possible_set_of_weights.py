import numpy as np
from itertools import product


def create_weights_list(min_value, max_value, step_size):
    return np.arange(min_value, max_value, step_size)


def get_weights_tentative(size, precision):
    possible_weights = create_weights_list(0, precision + 1, 1)
    tentative_weights = list(product(list(possible_weights), repeat=size))
    result = []
    for w in tentative_weights:
        if sum(w) == precision:
            result.append(list(w))
    return np.array(result) / precision, len(result)

def count_weights_tentative(size, precision):
    possible_weights = create_weights_list(0, precision + 1, 1)
    tentative_weights = list(product(list(possible_weights), repeat=size))
    count = 0
    for w in tentative_weights:
        if sum(w) == precision:
            count += 1
    return count

for i in range(8, 11):
    print('Generator size = {}, Possible set of weights = {}'.format(i, count_weights_tentative(i, 10)))