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

from typing import Dict, List

from prime_slam.slam.mapping.landmark.landmark import Landmark
from prime_slam.slam.mapping.map import Map

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
        return self._maps[landmark_name].descriptors

    def get_positions(self, landmark_name):
        return self._maps[landmark_name].positions

    def get_landmark_identifiers(self, landmark_name):
        return self._maps[landmark_name].landmark_identifiers

    def get_mean_viewing_directions(self, landmark_name):
        return self._maps[landmark_name].mean_viewing_directions

    def get_size(self, landmark_name):
        return len(self._maps[landmark_name])

    def get_landmarks(self, landmark_name):
        return self._maps[landmark_name].landmarks

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

    def cull_landmarks(self, landmark_name):
        self._maps[landmark_name].cull()
