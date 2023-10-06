from abc import ABC, abstractmethod


class SLAM(ABC):
    @abstractmethod
    def process_sensor_data(self, sensor_data):
        pass

    @property
    @abstractmethod
    def trajectory(self):
        pass

    @property
    @abstractmethod
    def map(self):
        pass
