from abc import ABC, abstractmethod
from typing import List

from prime_slam.observation.observation import Observation
from prime_slam.sensor.sensor_data import SensorData


class ObservationsFilter(ABC):
    @abstractmethod
    def apply(
        self, observations: List[Observation], sensor_data: SensorData
    ) -> List[Observation]:
        pass
