import numpy as np

from src.observation.keyobject import Keyobject


class Observation:
    def __init__(self, keyobject: Keyobject, descriptor: np.ndarray):
        self._keyobject = keyobject
        self._descriptor = descriptor

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def keyobject(self):
        return self._keyobject
