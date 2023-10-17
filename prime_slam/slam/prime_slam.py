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

from prime_slam.slam.backend.backend import Backend
from prime_slam.slam.frontend.frontend import Frontend
from prime_slam.slam.slam import SLAM


__all__ = ["PrimeSLAM"]


class PrimeSLAM(SLAM):
    def __init__(
        self,
        frontend: Frontend,
        backend: Backend,
    ):
        self.backend = backend
        self.frontend = frontend

    @property
    def trajectory(self):
        return self.frontend.trajectory

    @property
    def map(self):
        return self.frontend.map

    def process_sensor_data(self, sensor_data):
        new_frame = self.frontend.process_sensor_data(sensor_data)
        if new_frame.is_keyframe:
            for observation_name in new_frame.observations.observation_names:
                self.__optimize_graph(observation_name)

    def __optimize_graph(self, landmark_name):
        new_poses, new_landmark_positions = self.backend.optimize(self.frontend.graph)
        self.frontend.update_poses(new_poses)
        self.frontend.update_landmark_positions(new_landmark_positions, landmark_name)
