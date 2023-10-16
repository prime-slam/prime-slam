import numpy as np

from itertools import compress
from typing import List

from prime_slam.observation.filter.observation_filter import ObservationsFilter
from prime_slam.observation.observation import Observation
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["PointClipFilter"]


class PointClipFilter(ObservationsFilter):
    def apply(
        self, observations: List[Observation], sensor_data: SensorData
    ) -> List[Observation]:
        height, width = sensor_data.depth.depth_map.shape[:2]
        coordinates = np.array(
            [observation.keyobject.coordinates for observation in observations],
            dtype=int,
        )
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        return list(compress(observations, mask))
