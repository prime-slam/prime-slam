from typing import List, Dict

from src.frame import Frame
from src.mapping.map import Map
from src.mapping.landmark.landmark import Landmark


__all__ = ["MultiMap"]


class MultiMap:
    def __init__(self, landmarks: Dict[str, Dict[int, Landmark]] = None):
        self._maps: Dict[str, Map] = landmarks if landmarks is not None else {}

    @property
    def landmark_names(self):
        return list(self._maps.keys())

    def get_map(self, landmark_name):
        return self._maps[landmark_name]

    def add_map(self, landmark_name, new_map: Map):
        self._maps[landmark_name] = new_map

    def get_descriptors(self, landmark_name):
        return self._maps[landmark_name].get_descriptors()

    def get_positions(self, landmark_name):
        return self._maps[landmark_name].get_positions()

    def get_landmark_identifiers(self, landmark_name):
        return self._maps[landmark_name].get_landmark_identifiers()

    def get_mean_viewing_directions(self, landmark_name):
        return self._maps[landmark_name].get_mean_viewing_directions()

    def get_size(self, landmark_name):
        return self._maps[landmark_name].get_size()

    def get_landmarks(self, landmark_name):
        return self._maps[landmark_name].get_landmarks()

    def add_landmark(self, landmark: Landmark, landmark_name):
        if landmark_name not in self._maps:
            self._maps[landmark_name] = Map()
        self._maps[landmark_name].add_landmark(landmark)

    def add_landmarks(self, landmarks: List[Landmark], landmark_name):
        for landmark in landmarks:
            self.add_landmark(landmark, landmark_name)

    def add_associated_keyframe(self, landmark_name, landmark_id, keyframe: Frame):
        self._maps[landmark_name].add_associated_keyframe(landmark_id, keyframe)

    def update_position(self, new_positions, landmark_name):
        self._maps[landmark_name].update_position(new_positions)

    def recalculate_mean_viewing_directions(self, landmark_name):
        self._maps[landmark_name].recalculate_mean_viewing_directions()
