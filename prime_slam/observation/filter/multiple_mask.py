from typing import List

from prime_slam.observation.filter.observation_filter import ObservationsFilter
from prime_slam.observation.observation import Observation
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["MultipleFilter"]


class MultipleFilter(ObservationsFilter):
    def __init__(self, filters: List[ObservationsFilter]):
        self.observation_filters = filters

    def apply(
        self, observations: List[Observation], sensor_data: SensorData
    ) -> List[Observation]:
        result = self.observation_filters[0].apply(observations, sensor_data)
        for i in range(1, len(self.observation_filters)):
            result = self.observation_filters[i].apply(result, sensor_data)

        return result
