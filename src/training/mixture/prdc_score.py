import logging

import numpy as np
import torch
from helpers.configuration_container import ConfigurationContainer
from training.mixture.score_calculator import ScoreCalculator


class PRDCCalculator(ScoreCalculator):
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        cuda=True,
        verbose=False,
    ):
        pass
        # self.model = cargar MNISTCnn para conseguir los "emmbedings". Chequear si tiene softmax ya hecho

    def calculate(self, imgs, exact=True):
        pass
        # conseguir las clasificaciones para las reales y las fake
        # pasarle por el prdc
        # hacerle F1 al dc y dejarlo (f1, {})

    @property
    def is_reversed(self):
        return False
