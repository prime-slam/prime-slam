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

from typing import List


from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.mapping_config import MappingConfig
from prime_slam.slam.mapping.multi_map import MultiMap
from prime_slam.observation.keyobject import Keyobject
from prime_slam.utils.context_counter import ContextCounter

__all__ = ["Mapping"]


class Mapping:
    def __init__(self, mapping_configs: List[MappingConfig]):
        self.landmarks_counter = ContextCounter()
        self.mapping_configs = mapping_configs
        self.multi_map = MultiMap()

    def create_local_map(self, frame: Frame):
        local_multimap = MultiMap()
        for config in self.mapping_configs:
            observation_name = config.observation_name
            keyobjects: List[Keyobject] = frame.observations.get_keyobjects(
                observation_name
            )
            landmark_positions = config.projector.back_project(
                np.array([keyobject.coordinates for keyobject in keyobjects]),
                frame.sensor_measurement.depth.depth_map,
                frame.sensor_measurement.depth.intrinsics,
                frame.world_to_camera_transform,
            )
            descriptors = frame.observations.get_descriptors(observation_name)
            local_map = config.map_creator.create()
            for landmark_position, descriptor in zip(landmark_positions, descriptors):
                with self.landmarks_counter as current_id:
                    local_map.add_landmark(
                        local_map.create_landmark(
                            current_id,
                            landmark_position,
                            descriptor,
                            frame,
                        )
                    )

            local_multimap.add_map(local_map)
        return local_multimap

    def initialize_map(self, frame):
        local_map = self.create_local_map(frame)
        frame.local_map = local_map
        self.multi_map = local_map

    def add_associations(self, observation_name, frame, landmark_ids):
        for landmark_id in landmark_ids:
            self.multi_map.add_associated_keyframe(observation_name, landmark_id, frame)

    def add_landmark(self, landmark, observation_name):
        self.multi_map.add_landmark(
            landmark,
            observation_name,
        )

    def update_landmark_positions(self, new_positions, landmark_name):
        self.multi_map.update_positions(new_positions, landmark_name)
        self.multi_map.recalculate_mean_viewing_directions(landmark_name)

    def get_visible_multimap(
        self,
        frame: Frame,
    ):
        visible_multimap = MultiMap()
        for config in self.mapping_configs:
            observation_name = config.observation_name
            visible_map = self.multi_map.get_map(observation_name).get_visible_map(
                frame
            )
            visible_multimap.add_map(visible_map)

        return visible_multimap

    def cull_landmarks(self):
        for config in self.mapping_configs:
            observation_name = config.observation_name
            current_map = self.multi_map.get_map(observation_name)
            current_map.cull()
