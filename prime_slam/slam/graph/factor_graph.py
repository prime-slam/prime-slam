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

from typing import Dict

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.graph.factor.observation_factor import ObservationFactor
from prime_slam.slam.graph.node.landmark_node import LandmarkNode
from prime_slam.slam.graph.node.node import Node
from prime_slam.slam.graph.node.pose_node import PoseNode
from prime_slam.slam.mapping.landmark.landmark import Landmark

__all__ = ["FactorGraph"]


class FactorGraph:
    def __init__(self):
        self._pose_nodes = {}
        self._landmark_nodes = {}
        self._observation_factors = []

    @property
    def landmark_nodes(self):
        return list(self._landmark_nodes.values())

    @property
    def pose_nodes(self):
        return list(self._pose_nodes.values())

    @property
    def observation_factors(self):
        return self._observation_factors

    def add_pose_node(self, frame: Frame):
        self._pose_nodes[frame.identifier] = PoseNode(frame)

    def add_landmark_node(self, landmark: Landmark):
        self._landmark_nodes[landmark.identifier] = LandmarkNode(landmark)

    def add_observation_factor(
        self, pose_id, landmark_id, observation, sensor_measurement, information=None
    ):
        self._observation_factors.append(
            ObservationFactor(
                keyobject_position_2d=observation,
                pose_node=pose_id,
                landmark_node=landmark_id,
                sensor_measurement=sensor_measurement,
                information=information,
            )
        )

    def update(self):
        self._pose_nodes = self.__filter_bad_nodes(self._pose_nodes)
        self._landmark_nodes = self.__filter_bad_nodes(self._landmark_nodes)
        self._observation_factors = [
            factor
            for factor in self._observation_factors
            if factor.to_node in self._landmark_nodes
            and factor.from_node in self._pose_nodes
        ]

    @staticmethod
    def __filter_bad_nodes(nodes: Dict[int, Node]):
        return {node_id: node for node_id, node in nodes.items() if not node.is_bad}
