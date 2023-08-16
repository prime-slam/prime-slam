from abc import ABC, abstractmethod

from src.sensor.sensor_data_base import SensorData


class Detector(ABC):
    @abstractmethod
    def detect(self, sensor_data: SensorData):
        pass
