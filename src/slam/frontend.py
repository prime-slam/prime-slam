from abc import ABC, abstractmethod

from typing import Tuple

from src.data_association import DataAssociation
from src.frame import Frame
from src.mapping.map import Map
from src.sensor.sensor_data_base import SensorData


class Frontend(ABC):
    @abstractmethod
    def process_sensor_data(self, sensor_data: SensorData) -> Frame:
        pass

    # TODO: reorganize interface
    @abstractmethod
    def track(
        self,
        prev_frame: Frame,
        new_frame: Frame,
        sensor_data: SensorData,
    ) -> Frame:
        pass

    @abstractmethod
    def initialize_tracking(self, frame: Frame):
        pass

    @abstractmethod
    def add_new_keyframe(self, new_frame: Frame, matches_batch):
        pass
