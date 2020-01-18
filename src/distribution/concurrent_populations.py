from threading import Lock

from helpers.singleton import Singleton


@Singleton
class ConcurrentPopulations:

    def __init__(self):
        self._lock = Lock()
        self._generator = None
        self._discriminator = None

    def lock(self):
        self._lock.acquire()

    def unlock(self):
        try:
            self._lock.release()
        except RuntimeError:
            pass

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        self._generator = value

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value
