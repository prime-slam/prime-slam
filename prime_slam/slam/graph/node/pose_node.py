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

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.graph.node.node import Node
from prime_slam.slam.frame.frame import Frame

__all__ = ["PoseNode"]


class PoseNode(Node):
    def __init__(self, keyframe: Frame):
        super().__init__(keyframe.identifier)
        self._keyframe = keyframe

    @property
    def pose(self):
        return self._keyframe.world_to_camera_transform

    @property
    def is_bad(self):
        return not self._keyframe.is_keyframe
