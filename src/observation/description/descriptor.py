from abc import ABC, abstractmethod
from typing import List

from src.observation.keyobject import Keyobject
from src.sensor.sensor_data import SensorData
from src.typing.hints import ArrayNxM

__all__ = ["Descriptor"]


class Descriptor(ABC):
    @abstractmethod
    def descript(
        self,
        observations: List[Keyobject],
        sensor_data: SensorData,
    ) -> ArrayNxM[float]:
        pass
