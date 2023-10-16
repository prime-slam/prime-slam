import numpy as np

from itertools import compress
from typing import List

from prime_slam.observation.filter.observation_filter import ObservationsFilter
from prime_slam.observation.observation import Observation
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["PointDepthFilter"]


class PointDepthFilter(ObservationsFilter):
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
