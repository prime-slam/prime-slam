import numpy as np

from abc import ABC, abstractmethod
from typing import List

from src.observation.keyobject import Keyobject
from src.sensor.sensor_data_base import SensorData


class Descriptor(ABC):
    @abstractmethod
    def descript(
        self,
        observations: List[Keyobject],
        sensor_data: SensorData,
    ) -> np.ndarray:
        pass
