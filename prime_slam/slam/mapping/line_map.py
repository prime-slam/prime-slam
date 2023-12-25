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

from itertools import compress
from typing import List

from prime_slam.projection.projector import Projector
from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.landmark.landmark import Landmark
from prime_slam.slam.mapping.landmark.line_landmark import LineLandmark
from prime_slam.slam.mapping.map import Map

__all__ = ["LineMap"]


class LineMap(Map):
    def __init__(
        self, projector: Projector, landmark_name, landmarks: List[Landmark] = None
    ):
        super().__init__(projector, landmark_name, landmarks)

    def cull(self):
        for landmark in self._landmarks.values():
            landmark.update_state()
        self._landmarks = {
            landmark_id: landmark
            for landmark_id, landmark in self._landmarks.items()
            if not landmark.is_bad
        }

    def get_visible_map(
        self,
        frame: Frame,
    ) -> Map:
        visible_map = LineMap(self._projector, self._landmark_name)
        landmark_positions = self.positions
        landmark_positions_cam = self._projector.transform(
            landmark_positions, frame.world_to_camera_transform
        )
        projected_map = self._projector.project(
            landmark_positions_cam,
            frame.sensor_measurement.depth.intrinsics,
            np.eye(4),
        )
        depth_mask = (landmark_positions_cam[:, 2] > 0) & (
            landmark_positions_cam[:, 5] > 0
        )

        height, width = frame.sensor_measurement.depth.depth_map.shape[:2]
        x1 = projected_map[:, 0]
        y1 = projected_map[:, 1]
        x2 = projected_map[:, 2]
        y2 = projected_map[:, 3]
        mask = (
            (x1 >= 0)
            & (x1 < width)
            & (y1 >= 0)
            & (y1 < height)
            & (x2 >= 0)
            & (x2 < width)
            & (y2 >= 0)
            & (y2 < height)
            & depth_mask
        )
        visible_landmarks = list(compress(self._landmarks.values(), mask))
        visible_map.add_landmarks(visible_landmarks)
        return visible_map

    def create_landmark(self, current_id, landmark_position, descriptor, frame):
        return LineLandmark(current_id, landmark_position, descriptor, frame)
