from abc import abstractmethod, ABC


class ScoreCalculator(ABC):
    @abstractmethod
    def calculate(self, imgs, exact=True):
        pass

    @property
    @abstractmethod
    def is_reversed(self):
        pass
