from training.mixture.score_calculator import ScoreCalculator


class ConstantCalculator(ScoreCalculator):
    def __init__(self, cuda=False, resize=False):
        pass

    def calculate(self, imgs, exact=True):
        return 1, 0

    @property
    def is_reversed(self):
        return False
