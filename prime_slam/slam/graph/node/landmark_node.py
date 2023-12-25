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

from prime_slam.slam.mapping.landmark.landmark import Landmark
from prime_slam.slam.graph.node.node import Node
from prime_slam.slam.mapping.landmark.landmark import Landmark

__all__ = ["LandmarkNode"]


class LandmarkNode(Node):
    def __init__(self, landmark: Landmark):
        super().__init__(landmark.identifier)
        self._landmark = landmark

    @property
    def position(self):
        return self._landmark.position

    @property
    def is_bad(self):
        return self._landmark.is_bad
