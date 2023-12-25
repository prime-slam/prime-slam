# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from abc import ABC, abstractmethod
from typing import List

from prime_slam.geometry.util import normalize
from prime_slam.slam.frame.frame import Frame

__all__ = ["Landmark"]


class Landmark(ABC):
    def __init__(self, identifier, position, feature_descriptor, keyframe: Frame):
        self._identifier = identifier
        self._position = position
        self._feature_descriptor = feature_descriptor
        self._mean_viewing_direction = np.array([0, 0, 0], dtype=float)
        self._associated_keyframes: List[Frame] = []
        self.add_associated_keyframe(keyframe)
        self.keyframes_from_last_insert = 0
        self._is_bad = False

    def add_associated_keyframe(self, keyframe: Frame):
        self._associated_keyframes.append(keyframe)
        self.__add_viewing_direction(keyframe)

    def recalculate_mean_viewing_direction(self):
        origins = np.array([kf.origin for kf in self._associated_keyframes])
        viewing_directions = np.array(
            [
                normalize(viewing_direction)
                for viewing_direction in self._calculate_viewing_directions(origins)
            ]
        )
        self._mean_viewing_direction = normalize(
            np.sum(
                viewing_directions,
                axis=0,
            )
        )

    @property
    def is_bad(self):
        return self._is_bad

    @property
    def is_bad(self):
        return self._is_bad

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

    def update_state(self):
        self.keyframes_from_last_insert += 1
        if self.keyframes_from_last_insert >= 2:
            self._is_bad = len(self._associated_keyframes) < 2

    def __add_viewing_direction(self, associated_keyframe):
        viewing_direction = self._calculate_viewing_directions(
            associated_keyframe.origin
        )
        self._mean_viewing_direction += normalize(viewing_direction)
        self._mean_viewing_direction = normalize(self._mean_viewing_direction)

    @abstractmethod
    def _calculate_viewing_directions(self, origins: np.ndarray):
        pass
