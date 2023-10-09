from abc import ABC, abstractmethod
from typing import List

from src.observation.keyobject import Keyobject
from src.sensor.sensor_data import SensorData

__all__ = ["Detector"]


class Detector(ABC):
    @abstractmethod
    def detect(self, sensor_data: SensorData) -> List[Keyobject]:
        pass
