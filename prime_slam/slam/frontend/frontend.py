# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.graph.factor_graph import FactorGraph
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["Frontend"]


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

    @abstractmethod
    def update_graph(self):
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
