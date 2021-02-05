#!/usr/bin/env python3
import logging

from training.mixture.fid_score import FIDCalculator
from training.mixture.prdc_score import PRDCCalculator
from training.mixture.score_calculator import ScoreCalculator


class CombinedCalculator(ScoreCalculator):

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        imgs_original,
        batch_size=64,
        dims=2048,
        n_samples=10000,
        cuda=True,
        verbose=False,
        nearest_k=5,
        use_random_vgg=False,
    ):
        """
        :param imgs_original: The original dataset, e.g. torcvision.datasets.CIFAR10
        :param batch_size: Batch size that will be used, 64 is recommended.
        :param cuda: If True, the GPU will be used.
        :param dims: Dimensionality of Inception features to use. By default, uses pool3 features.
        :param n_samples: In the paper, min. 10k samples are suggested.
        :param verbose: Verbose logging
        """
        self.fid_calculator = FIDCalculator(
            imgs_original,
            batch_size=batch_size,
            dims=dims,
            n_samples=n_samples,
            cuda=cuda,
            verbose=verbose,
        )
        self.prdc_calculator = PRDCCalculator(
            imgs_original,
            batch_size=batch_size,
            dims=dims,
            n_samples=n_samples,
            cuda=cuda,
            verbose=verbose,
            nearest_k=nearest_k,
            use_random_vgg=use_random_vgg,
        )

    def calculate(self, imgs, exact=True):
        fid, tvd = self.fid_calculator.calculate(imgs)
        _, prdc_dict = self.prdc_calculator.calculate(imgs)
        precision = prdc_dict["precision"]
        recall = prdc_dict["recall"]
        density = prdc_dict["density"]
        coverage = prdc_dict["coverage"]

        # return values ordered by importance in final tuple for sorting.
        return fid, tvd, coverage, density, recall, precision

    @property
    def is_reversed(self):
        return True
