import numpy as np

from abc import ABC, abstractmethod
from typing import List

from prime_slam.slam.frame.frame import Frame
from prime_slam.geometry.util import normalize

__all__ = ["Landmark"]


class Landmark(ABC):
    def __init__(self, identifier, position, feature_descriptor, keyframe: Frame):
        self._identifier = identifier
        self._position = position
        self._feature_descriptor = feature_descriptor
        self._mean_viewing_direction = np.array([0, 0, 0], dtype=float)
        self._associated_keyframes: List[Frame] = []
        self.add_associated_keyframe(keyframe)

    def add_associated_keyframe(self, keyframe: Frame):
        self._associated_keyframes.append(keyframe)
        self.__add_viewing_direction(keyframe)

    def recalculate_mean_viewing_direction(self):
        origins = np.array([kf.origin for kf in self._associated_keyframes])
        self._mean_viewing_direction = np.sum(
            normalize(self._calculate_viewing_directions(origins)),
            axis=0,
        )
        self._mean_viewing_direction = normalize(self._mean_viewing_direction)

    @property
    def identifier(self):
        return self._identifier

    @property
    def associated_keyframes(self):
        return self._associated_keyframes

    @property
    def position(self):
        return self._position

    @property
    def feature_descriptor(self):
        return self._feature_descriptor

    @position.setter
    def position(self, new_position):
        self._position = new_position

    @property
    def mean_viewing_direction(self):
        return self._mean_viewing_direction

    def __add_viewing_direction(self, associated_keyframe):
        viewing_direction = self._calculate_viewing_directions(
            associated_keyframe.origin
        )
        viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)
        self._mean_viewing_direction += viewing_direction
        self._mean_viewing_direction = normalize(self._mean_viewing_direction)

    @abstractmethod
    def _calculate_viewing_directions(self, origins: np.ndarray):
        pass
