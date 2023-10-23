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

from prime_slam.typing.hints import Array3x3, Array4x4, Array3


__all__ = ["make_euclidean_transform", "make_homogeneous_matrix"]


def make_euclidean_transform(
    rotation_matrix: Array3x3[float], translation: Array3[float]
) -> Array4x4[float]:
    transform = np.hstack([rotation_matrix, translation.reshape(-1, 1)])
    transform = np.vstack([transform, [0, 0, 0, 1]])
    return transform


def make_homogeneous_matrix(matrix: Array3x3[float]) -> Array4x4[float]:
    matrix_homo = np.hstack([matrix, np.zeros((3, 1))])
    matrix_homo = np.vstack([matrix_homo, [0, 0, 0, 1]])
    return matrix_homo
