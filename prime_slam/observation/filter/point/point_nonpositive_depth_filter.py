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

from prime_slam.observation.filter.observation_filter import ObservationsFilter
from prime_slam.observation.observation import Observation
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["PointNonpositiveDepthFilter"]


class PointNonpositiveDepthFilter(ObservationsFilter):
    def apply(
        self, observations: List[Observation], sensor_data: SensorData
    ) -> List[Observation]:
        coordinates = np.array(
            [observation.keyobject.coordinates for observation in observations],
            dtype=int,
        )
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        mask = sensor_data.depth.depth_map[y, x] > 0

        return list(compress(observations, mask))
