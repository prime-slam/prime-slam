import numpy as np
from typing import List

from src.observation.keyobject import Keyobject
from src.observation.observation import Observation


class ObservationsBatch:
    def __init__(
        self,
        keyobjects_batch: List[List[Keyobject]],
        descriptors_batch: List[np.ndarray],
        names: List[str],
    ):
        self._observations_batch = {}
        self._descriptors_batch = {}
        self._keyobjects_batch = {}
        self._names = names
        for keyobjects, descriptors, name in zip(
            keyobjects_batch, descriptors_batch, names
        ):
            self._observations_batch[name] = [
                Observation(keyobject, descriptor)
                for keyobject, descriptor in zip(keyobjects, descriptors)
            ]
            self._descriptors_batch[name] = descriptors
            self._keyobjects_batch[name] = keyobjects

    @property
    def observation_names(self):
        return self._names

    def get_size(self, name):
        return len(self._observations_batch[name])

    def get_observations(self, name):
        return self._observations_batch[name]

    def get_descriptors(self, name):
        return self._descriptors_batch[name]

    def get_keyobjects(self, name):
        return self._keyobjects_batch[name]
