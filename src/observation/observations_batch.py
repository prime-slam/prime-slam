import numpy as np
from typing import List

from src.observation.keyobject import Keyobject
from src.observation.observation import Observation


class ObservationsBatch:
    def __init__(self, keyobjects: List[Keyobject], descriptors: np.ndarray):
        self._observations = [
            Observation(keyobject, descriptor)
            for keyobject, descriptor in zip(keyobjects, descriptors)
        ]
        self._descriptors = descriptors
        self._keyobjects = keyobjects

    @property
    def observations(self):
        return self._observations

    @property
    def descriptors(self):
        return self._descriptors

    @property
    def keyobjects(self):
        return self._keyobjects
