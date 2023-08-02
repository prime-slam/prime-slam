import numpy as np

from abc import ABC, abstractmethod
from typing import List

from src.feature.feature import Feature
from src.sensor.sensor_data_base import SensorData


class Matcher(ABC):
    @abstractmethod
    def match(
        self,
        first_features: List[Feature],
        second_features: List[Feature],
        first_sensor_data: SensorData,
        second_sensor_data: SensorData,
    ) -> np.ndarray:
        pass
