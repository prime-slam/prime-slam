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

import numpy as np

from typing import Optional

from prime_slam.geometry.transform import make_euclidean_transform
from prime_slam.typing.hints import Rotation, Translation, Transformation

__all__ = ["Pose"]


class Pose:
    def __init__(self, transformation: Optional[Transformation] = None):
        self._transform: Transformation = None
        self._transform_inv: Transformation = None
        self._rotation = None
        self._translation = None
        self._rotation_inv = None
        self._translation_inv = None
        if transformation is not None:
            self.update(transformation)

    def update(self, new_transformation: Transformation) -> None:
        self._transform: Transformation = new_transformation
        self._transform_inv: Transformation = np.linalg.inv(self._transform)
        self._rotation = self._transform[:3, :3]
        self._translation = self._transform[:3, 3]

    def inverse(self) -> "Pose":
        return Pose(np.linalg.inv(self._transform))

    @property
    def transformation(self) -> Transformation:
        return self._transform

    @property
    def rotation(self) -> Rotation:
        return self._rotation

    @property
    def translation(self) -> Translation:
        return self._translation

    @classmethod
    def from_rotation_and_translation(
        cls, rotation: Rotation, translation: Translation
    ) -> "Pose":
        return cls(make_euclidean_transform(rotation, translation))
