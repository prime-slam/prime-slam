from typing import List, Dict

from prime_slam.slam.mapping.map import Map
from prime_slam.slam.mapping.landmark.landmark import Landmark


__all__ = ["MultiMap"]


class MultiMap:
    def __init__(self, maps: List[Map] = None):
        self._maps: Dict[str, Map] = {}
        if maps is not None:
            self.add_maps(maps)

    @property
    def landmark_names(self):
        return list(self._maps.keys())

    def get_map(self, landmark_name):
        return self._maps[landmark_name]

    def add_map(self, new_map: Map):
        self._maps[new_map.name] = new_map

    def add_maps(self, maps: List[Map]):
        for new_map in maps:
            self.add_map(new_map)

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
        self._maps[landmark_name].add_landmark(landmark)

    def add_landmarks(self, landmarks: List[Landmark], landmark_name):
        for landmark in landmarks:
            self.add_landmark(landmark, landmark_name)

    def add_associated_keyframe(self, landmark_name, landmark_id, keyframe):
        self._maps[landmark_name].add_associated_keyframe(landmark_id, keyframe)

    def update_positions(self, new_positions, landmark_name):
        self._maps[landmark_name].update_position(new_positions)

    def recalculate_mean_viewing_directions(self, landmark_name):
        self._maps[landmark_name].recalculate_mean_viewing_directions()
