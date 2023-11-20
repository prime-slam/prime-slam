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

from abc import ABC, abstractmethod

__all__ = ["Projector"]


class Projector(ABC):
    @abstractmethod
    def transform(self, objects_3d, transformation_matrix):
        pass

    @abstractmethod
    def project(
        self,
        object_3d,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        pass

    @abstractmethod
    def back_project(
        self,
        object_2d,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        pass
