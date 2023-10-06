from typing import List, Dict

import numpy as np

from src.mapping.landmark import Landmark


class Map:
    def __init__(self, landmarks: Dict[str, Dict[int, Landmark]] = None):
        self._landmarks = landmarks if landmarks is not None else {}

    @property
    def landmark_names(self):
        return list(self._landmarks.keys())

    def get_descriptors(self, landmark_name):
        return np.array(
            [
                landmark.feature_descriptor
                for landmark in self._landmarks[landmark_name].values()
            ]
        )

    def get_positions(self, landmark_name):
        return np.array(
            [landmark.position for landmark in self._landmarks[landmark_name].values()]
        )

    def get_landmark_identifiers(self, landmark_name):
        return self._landmarks[landmark_name].keys()

    def get_mean_viewing_directions(self, landmark_name):
        return np.array(
            [
                landmark.mean_viewing_direction
                for landmark in self._landmarks[landmark_name].values()
            ]
        )

    def get_size(self, landmark_name):
        return len(self._landmarks[landmark_name])

    def get_landmarks(self, landmark_name):
        return list(self._landmarks[landmark_name].values())

    def add_landmark(self, landmark: Landmark, landmark_name):
        if landmark_name not in self._landmarks:
            self._landmarks[landmark_name] = {}
        self._landmarks[landmark_name][landmark.identifier] = landmark

    def add_landmarks(self, landmarks: List[Landmark], landmark_name):
        for landmark in landmarks:
            self.add_landmark(landmark, landmark_name)

    def add_associated_keyframe(self, landmark_name, landmark_id, keyframe):
        self._landmarks[landmark_name][landmark_id].add_associated_keyframe(keyframe)

    def update_position(self, new_positions, landmark_name):
        for landmark, new_position in zip(
            self._landmarks[landmark_name].values(), new_positions
        ):
            landmark.position = new_position

    def recalculate_mean_viewing_directions(self, landmark_name):
        for landmark in self._landmarks[landmark_name].values():
            landmark.recalculate_mean_viewing_direction()
