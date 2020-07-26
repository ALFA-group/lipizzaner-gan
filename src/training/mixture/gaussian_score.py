import logging

from training.mixture.score_calculator import ScoreCalculator
from scipy.spatial import distance
import numpy as np


class GaussianToyDistancesCalculator(ScoreCalculator):
    _logger = logging.getLogger(__name__)

    def __init__(self, centers, batch_size=64, dims=2048, n_samples=1000, cuda=True, verbose=False):
        xs, ys = centers
        self.centers = np.array((xs, ys), dtype=np.float).T
        self.n_samples = n_samples

    def calculate(self, imgs, exact=True):
        distances = []
        classes = len(self.centers) * [0]
        for i in range(self.n_samples):
            _distance = distance.cdist([imgs[i][0:2].numpy()], self.centers, 'euclidean')
            min_distance = np.amin(_distance)
            mode_class = np.where(_distance == min_distance)
            distances.append(min_distance)
            classes[mode_class[1][0]] += 1

        real_data_frequencies = 1/len(self.centers)
        classes = np.array(classes)/self.n_samples
        tvd = 0.5 * sum(abs(real_data_frequencies - fake_data_frequency) for fake_data_frequency in classes)
        return np.mean(distances), tvd

    @property
    def is_reversed(self):
        return False
