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

from prime_slam.geometry.pose import Pose
from prime_slam.observation.observations_batch import ObservationsBatch
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["Frame"]


class Frame:
    def __init__(
        self,
        observations: ObservationsBatch,
        sensor_measurement: SensorData,
        local_map=None,
        world_to_camera_transform: Pose = None,
        is_keyframe: bool = False,
        identifier: int = 0,
    ):
        self.observations: ObservationsBatch = observations
        self.sensor_measurement = sensor_measurement
        self.local_map = local_map
        self._world_to_camera_transform: Pose = world_to_camera_transform
        self._camera_to_world_transform: Pose = (
            world_to_camera_transform.inverse()
            if world_to_camera_transform is not None
            else None
        )
        self._is_keyframe = is_keyframe
        self.identifier = identifier

    def __hash__(self):
        return self.identifier

    @property
    def is_keyframe(self):
        return self._is_keyframe

    @is_keyframe.setter
    def is_keyframe(self, value):
        self._is_keyframe = value

    def update_pose(self, new_pose: Pose):
        self._world_to_camera_transform = new_pose
        self._camera_to_world_transform = new_pose.inverse()

    @property
    def world_to_camera_transform(self):
        return self._world_to_camera_transform.transformation

    @property
    def camera_to_world_transform(self):
        return self._camera_to_world_transform.transformation

    @property
    def world_to_camera_rotation(self):
        return self._world_to_camera_transform.rotation

    @property
    def camera_to_world_rotation(self):
        return self._camera_to_world_transform.rotation

    @property
    def world_to_camera_translation(self):
        return self._world_to_camera_transform.translation

    @property
    def camera_to_world_translation(self):
        return self._camera_to_world_transform.translation

    @property
    def origin(self):
        return -(self.world_to_camera_rotation @ self.camera_to_world_translation)
