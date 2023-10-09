import numpy as np

from src.observation.keyobject import Keyobject

__all__ = ["Keyline"]


class Keyline(Keyobject):
    def __init__(self, x1, y1, x2, y2, uncertainty=None):
        self._coordinates = np.array([x1, y1, x2, y2])
        self._uncertainty = uncertainty

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def uncertainty(self):
        return self._uncertainty

    @property
    def data(self):
        return self._coordinates
