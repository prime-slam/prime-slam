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

from prime_slam.typing.hints import ArrayN, ArrayNx4

__all__ = ["clip_lines", "normalize"]


def clip_lines(lines: ArrayNx4[float], height: float, width: float) -> ArrayNx4[float]:
    x_index = [0, 2]
    lines[..., x_index] = np.clip(lines[..., x_index], 0, width - 1)
    y_index = [1, 3]
    lines[..., y_index] = np.clip(lines[..., y_index], 0, height - 1)
    return lines


def normalize(vector: ArrayN[float], epsilon: float = 1.0e-10) -> ArrayN[float]:
    norm = np.linalg.norm(vector)
    return vector / norm if norm >= epsilon else vector
