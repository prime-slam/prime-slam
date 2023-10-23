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

from itertools import compress
from typing import List

import numpy as np

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.landmark.landmark import Landmark

from prime_slam.slam.mapping.landmark.point_landmark import PointLandmark
from prime_slam.slam.mapping.map import Map
from prime_slam.projection.projector import Projector

__all__ = ["PointMap"]


class PointMap(Map):
    def __init__(
        self, projector: Projector, landmark_name, landmarks: List[Landmark] = None
    ):
        super().__init__(projector, landmark_name, landmarks)

    def get_visible_map(
        self,
        frame: Frame,
    ):
        visible_map = PointMap(self._projector, self._landmark_name)
        landmark_positions = self.get_positions()
        landmark_positions_cam = self._projector.transform(
            landmark_positions, frame.world_to_camera_transform
        )
        map_mean_viewing_directions = self.get_mean_viewing_directions()
        projected_map = self._projector.project(
            landmark_positions_cam,
            frame.sensor_measurement.depth.intrinsics,
            np.eye(4),
        )
        depth_mask = landmark_positions_cam[:, 2] > 0
        origin = frame.origin
        viewing_directions = landmark_positions - origin
        viewing_directions = viewing_directions / np.linalg.norm(
            viewing_directions, axis=-1
        ).reshape(-1, 1)
        height, width = frame.sensor_measurement.depth.depth_map.shape[:2]

        viewing_direction_mask = (
            np.sum(map_mean_viewing_directions * viewing_directions, axis=-1) >= 0.5
        )
        mask = (
            (projected_map[:, 0] >= 0)
            & (projected_map[:, 0] < width)
            & (projected_map[:, 1] >= 0)
            & (projected_map[:, 1] < height)
            & depth_mask
            # & viewing_direction_mask
        )
        visible_landmarks = list(compress(self._landmarks.values(), mask))
        visible_map.add_landmarks(visible_landmarks)

        return visible_map

    def create_landmark(self, current_id, landmark_position, descriptor, frame):
        return PointLandmark(current_id, landmark_position, descriptor, frame)
