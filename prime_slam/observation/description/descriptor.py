from abc import ABC, abstractmethod
from typing import List

from prime_slam.observation.keyobject import Keyobject
from prime_slam.sensor.sensor_data import SensorData
from prime_slam.typing.hints import ArrayNxM

__all__ = ["Descriptor"]


class Descriptor(ABC):
    @abstractmethod
    def descript(
        self,
        observations: List[Keyobject],
        sensor_data: SensorData,
    ) -> ArrayNxM[float]:
        pass
