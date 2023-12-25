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

from prime_slam.slam.mapping.map_creator.map_creator import MapCreator
from prime_slam.slam.mapping.point_map import PointMap

__all__ = ["PointMapCreator"]


class PointMapCreator(MapCreator):
    def __init__(self, projector, landmark_name: str):
        super().__init__(projector, landmark_name)

    def create(self):
        return PointMap(self.projector, self.landmark_name)
