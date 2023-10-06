from abc import ABC, abstractmethod, abstractproperty

from src.frame import Frame
from src.graph.factor_graph import FactorGraph
from src.mapping.map import Map
from src.sensor.sensor_data_base import SensorData


class Frontend(ABC):
    @abstractmethod
    def process_sensor_data(self, sensor_data: SensorData) -> Frame:
        pass

    @abstractmethod
    def update_poses(self, new_poses):
        pass

    @abstractmethod
    def update_landmark_positions(self, new_positions, landmark_name):
        pass

    @property
    @abstractmethod
    def graph(self) -> FactorGraph:
        pass

    @property
    @abstractmethod
    def map(self) -> FactorGraph:
        pass

    @property
    @abstractmethod
    def trajectory(self) -> FactorGraph:
        pass
