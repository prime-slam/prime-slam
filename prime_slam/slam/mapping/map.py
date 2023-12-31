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

from prime_slam.projection.projector import Projector
from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.landmark.landmark import Landmark

__all__ = ["Map"]


class Map(ABC):
    def __init__(
        self,
        projector: Projector,
        landmark_name: str,
        landmarks: List[Landmark] = None,
    ):
        self._projector = projector
        self._landmarks = {}
        self._landmark_name = landmark_name
        if landmarks is not None:
            self.add_landmarks(landmarks)

    @property
    def name(self):
        return self._landmark_name

    @property
    def descriptors(self):
        return np.array(
            [landmark.feature_descriptor for landmark in self._landmarks.values()]
        )

    @property
    def positions(self):
        return np.array([landmark.position for landmark in self._landmarks.values()])

    @property
    def landmark_identifiers(self):
        return self._landmarks.keys()

    @property
    def mean_viewing_directions(self):
        return np.array(
            [landmark.mean_viewing_direction for landmark in self._landmarks.values()]
        )

    def __len__(self):
        return len(self._landmarks)

    @property
    def landmarks(self):
        return list(self._landmarks.values())

    def add_landmark(self, landmark: Landmark):
        self._landmarks[landmark.identifier] = landmark

    def add_landmarks(self, landmarks: List[Landmark]):
        for landmark in landmarks:
            self.add_landmark(landmark)

    def add_associated_keyframe(self, landmark_id, keyframe: Frame):
        self._landmarks[landmark_id].add_associated_keyframe(keyframe)

    def update_position(self, new_positions):
        for landmark, new_position in zip(self._landmarks.values(), new_positions):
            landmark.position = new_position

    def recalculate_mean_viewing_directions(self):
        for landmark in self._landmarks.values():
            landmark.recalculate_mean_viewing_direction()

    @abstractmethod
    def get_visible_map(
        self,
        frame: Frame,
    ):
        pass

    @abstractmethod
    def create_landmark(self, current_id, landmark_position, descriptor, frame):
        pass

    @abstractmethod
    def cull(self):
        pass
