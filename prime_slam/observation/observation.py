import numpy as np

from prime_slam.observation.keyobject import Keyobject

__all__ = ["Observation"]


class Observation:
    def __init__(self, keyobject: Keyobject, descriptor: np.ndarray):
        self._keyobject = keyobject
        self._descriptor = descriptor

    @property
    def descriptor(self) -> np.ndarray:
        return self._descriptor

    @property
    def keyobject(self) -> Keyobject:
        return self._keyobject
