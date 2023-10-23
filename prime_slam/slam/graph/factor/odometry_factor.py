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

from prime_slam.slam.graph.factor.factor import Factor

__all__ = ["OdometryFactor"]


class OdometryFactor(Factor):
    def __init__(self, relative_pose, from_node, to_node, information=None):
        super().__init__(from_node, to_node, information)
        self._relative_pose = relative_pose

    @property
    def relative_pose(self):
        return self._relative_pose

    @relative_pose.setter
    def relative_pose(self, new_relative_pose):
        self._relative_pose = new_relative_pose
