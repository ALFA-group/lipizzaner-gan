import logging
import torch

from helpers.configuration_container import ConfigurationContainer
from training.mixture.score_calculator import ScoreCalculator
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import numpy as np

_logger = logging.getLogger()
class GaussianToyDistancesCalculator2D(ScoreCalculator):

    def __init__(
        self,
        centers,
        batch_size=64,
        dims=2048,
        n_samples=1000,
        cuda=True,
        verbose=False,
    ):
        xs, ys, self.labels = centers
        self.centers = np.array((xs, ys), dtype=np.float).T
        self.n_samples = n_samples

    def calculate(self, imgs, exact=True):
        distances = []
        classes = len(self.centers) * [0]

        for i in range(self.n_samples):
            _distance = distance.cdist([imgs[i][0:2].numpy()], self.centers, "euclidean")
            min_distance = np.amin(_distance)
            mode_class = np.where(_distance == min_distance)
            distances.append(min_distance)
            classes[mode_class[1][0]] += 1

        real_data_frequencies = 1 / len(self.centers)
        classes = np.array(classes) / self.n_samples
        tvd = 0.5 * sum(abs(real_data_frequencies - fake_data_frequency) for fake_data_frequency in classes)
        return np.mean(distances), tvd

    @property
    def is_reversed(self):
        return True


class GaussianToyDistancesCalculator1D(ScoreCalculator):

    def __init__(
        self,
        target_distribution=None,
        score_sample_size=10000,
        cuda=True,
        verbose=False,
    ):
        self.cc = ConfigurationContainer.instance()
        self.score_sample_size = self.cc.settings["trainer"]["params"]["score"].get("sample_size", score_sample_size)
        _logger.info('target distribution is {} and type is {}'.format(target_distribution[0], type(target_distribution[0])))
        array = target_distribution[0].cpu().data.numpy() # np.array(target_distribution[0].cpu().data) # np.array()
        self.target_distribution = array.reshape(1, -1)  # [0] are the samples

    def calculate(self, imgs, exact=True):  # The score is the Wasserstein distance between both distributions
        samples = list()
        for i in range(len(imgs)):
            samples.append(imgs[i])
        imgs = np.squeeze(np.array(torch.cat(samples)))
        return wasserstein_distance(imgs, self.target_distribution[0]), 0

    @property
    def is_reversed(self):
        return True
