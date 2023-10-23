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

from prime_slam.slam.graph.factor.factor import Factor
from prime_slam.sensor.rgbd import RGBDImage

__all__ = ["ObservationFactor"]


class ObservationFactor(Factor):
    def __init__(
        self,
        keyobject_position_2d,
        pose_node,
        landmark_node,
        sensor_measurement: RGBDImage,
        information=None,
    ):
        super().__init__(
            pose_node,
            landmark_node,
            information if information is not None else np.eye(2),
        )
        self._keyobject_2d = keyobject_position_2d
        self.depth_map = sensor_measurement.depth.depth_map
        self.depth_scale = sensor_measurement.depth.depth_scale

    @property
    def observation(self):
        return self._keyobject_2d
