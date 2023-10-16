import numpy as np
from typing import List

from prime_slam.observation.keyobject import Keyobject
from prime_slam.observation.observation import Observation

__all__ = ["ObservationsBatch"]


class ObservationsBatch:
    def __init__(
        self,
        observations_batch: List[List[Observation]],
        names: List[str],
    ):
        self._observations_batch = {}
        self._descriptors_batch = {}
        self._keyobjects_batch = {}
        self._names = names
        for observations, name in zip(observations_batch, names):
            self._observations_batch[name] = observations
            self._descriptors_batch[name] = np.array(
                [observation.descriptor for observation in observations]
            )
            self._keyobjects_batch[name] = [
                observation.keyobject for observation in observations
            ]

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
