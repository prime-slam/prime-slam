import numpy as np

from typing import List

from src.frame import Frame

__all__ = ["Landmark"]


class Landmark:
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
            self.__normalize(self._position - origins), axis=0
        )
        self.__normalize_mean_viewing_direction()

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
        viewing_direction = self._position - associated_keyframe.origin
        viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)
        self._mean_viewing_direction += viewing_direction
        self.__normalize_mean_viewing_direction()

    def __normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm >= 1.0e-10:
            vector /= norm
        return vector

    def __normalize_mean_viewing_direction(self):
        norm = np.linalg.norm(self._mean_viewing_direction)
        if norm >= 1.0e-10:
            self._mean_viewing_direction /= np.linalg.norm(self._mean_viewing_direction)
