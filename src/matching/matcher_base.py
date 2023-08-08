import numpy as np

from abc import ABC, abstractmethod
from typing import List

from src.observation.observation import Observation
from src.sensor.sensor_data_base import SensorData


class Matcher(ABC):
    @abstractmethod
    def match(
        self,
        first_observations: List[Observation],
        second_observations: List[Observation],
        first_sensor_data: SensorData,
        second_sensor_data: SensorData,
    ) -> np.ndarray:
        pass
