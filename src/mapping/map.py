from typing import List, Dict

import numpy as np

from src.frame import Frame
from src.mapping.landmark.landmark import Landmark

__all__ = ["Map"]


class Map:
    def __init__(self, landmarks: Dict[int, Landmark] = None):
        self._landmarks = landmarks if landmarks is not None else {}

    def get_descriptors(self):
        return np.array(
            [landmark.feature_descriptor for landmark in self._landmarks.values()]
        )

    def get_positions(self):
        return np.array([landmark.position for landmark in self._landmarks.values()])

    def get_landmark_identifiers(self):
        return self._landmarks.keys()

    def get_mean_viewing_directions(self):
        return np.array(
            [landmark.mean_viewing_direction for landmark in self._landmarks.values()]
        )

    def get_size(self):
        return len(self._landmarks)

    def get_landmarks(self):
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
