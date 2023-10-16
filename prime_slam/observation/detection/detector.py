from abc import ABC, abstractmethod
from typing import List

from prime_slam.observation.keyobject import Keyobject
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["Detector"]


class Detector(ABC):
    @abstractmethod
    def detect(self, sensor_data: SensorData) -> List[Keyobject]:
        pass
