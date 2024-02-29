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

from prime_slam.typing.hints import Transformation

__all__ = ["TUM_DEPTH_FACTOR", "TUM_DEFAULT_INTRINSICS", "ICL_NUIM_DEFAULT_INTRINSICS"]

TUM_DEPTH_FACTOR: float = 5000

TUM_DEFAULT_INTRINSICS: Transformation = np.array(
    [
        [525.0, 0, 319.5, 0],
        [0, 525.0, 239.5, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

ICL_NUIM_DEFAULT_INTRINSICS = np.array(
    [
        [481.2, 0, 319.5, 0],
        [0, -480.0, 239.5, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
