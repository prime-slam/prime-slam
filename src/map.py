from typing import List, Dict

import numpy as np

from src.landmark import Landmark


class Map:
    def __init__(self, landmarks: Dict[str, List[Landmark]] = None):
        self._landmarks = landmarks if landmarks is not None else {}

    def get_descriptors(self, landmark_name):
        return np.array(
            [landmark.feature_descriptor for landmark in self._landmarks[landmark_name]]
        )

    def get_positions(self, landmark_name):
        return np.array(
            [landmark.position for landmark in self._landmarks[landmark_name]]
        )

    def get_mean_viewing_directions(self, landmark_name):
        return np.array(
            [
                landmark.mean_viewing_direction
                for landmark in self._landmarks[landmark_name]
            ]
        )

    def get_size(self, landmark_name):
        return len(self._landmarks[landmark_name])

    def get_landmarks(self, landmark_name):
        return self._landmarks[landmark_name]

    def add_landmark(self, landmark: Landmark, landmark_name):
        if landmark_name not in self._landmarks:
            self._landmarks[landmark_name] = []
        self._landmarks[landmark_name].append(landmark)

    def add_landmarks(self, landmarks: List[Landmark], landmark_name):
        if landmark_name not in self._landmarks:
            self._landmarks[landmark_name] = []
        self._landmarks[landmark_name].extend(landmarks)

    def add_viewing_direction(self, landmark_name, landmark_id, viewing_direction):
        self._landmarks[landmark_name][landmark_id].add_viewing_direction(
            viewing_direction
        )

    def update_position(self, new_positions, landmark_name):
        for landmark, new_position in zip(
            self._landmarks[landmark_name], new_positions
        ):
            landmark.position = new_position
