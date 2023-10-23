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

from typing import List

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.landmark.landmark import Landmark
from prime_slam.slam.mapping.landmark.line_landmark import LineLandmark
from prime_slam.slam.mapping.map import Map
from prime_slam.projection.projector import Projector

__all__ = ["LineMap"]


class LineMap(Map):
    def __init__(
        self, projector: Projector, landmark_name, landmarks: List[Landmark] = None
    ):
        super().__init__(projector, landmark_name, landmarks)

    def get_visible_map(
        self,
        frame: Frame,
    ) -> Map:
        raise NotImplementedError()

    def create_landmark(self, current_id, landmark_position, descriptor, frame):
        return LineLandmark(current_id, landmark_position, descriptor, frame)
