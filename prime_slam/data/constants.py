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

__all__ = [
    "TUM_DEPTH_FACTOR",
    "TUM_DEFAULT_INTRINSICS",
    "ICL_NUIM_DEFAULT_INTRINSICS",
    "HILTI_DEFAULT_INTRINSICS_CAM0",
    "HILTI_DEFAULT_INTRINSICS_CAM1",
    "HILTI_DISTORTION_CAM0",
    "HILTI_DISTORTION_CAM1",
    "HILTI_IMAGE_SIZE",
    "HILTI_TRANSFORM_CAM0_TO_CAM1",
]

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

HILTI_DEFAULT_INTRINSICS_CAM0 = np.array(
    [
        [351.31400364193297, 0, 367.8522793375995, 0],
        [0, 351.4911744656785, 253.8402144980996, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

HILTI_DEFAULT_INTRINSICS_CAM1 = np.array(
    [
        [352.6489794433894, 0, 347.8170010310082, 0],
        [0, 352.8586498571586, 270.5806692485468, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

HILTI_DISTORTION_CAM0 = np.array(
    [
        -0.03696737352869157,
        -0.008917880497032812,
        0.008912969593422046,
        -0.0037685977496087313,
    ]
)

HILTI_DISTORTION_CAM1 = np.array(
    [
        -0.039086652082708805,
        -0.005525347047415151,
        0.004398151558986798,
        -0.0019701263170917808,
    ]
)

HILTI_TRANSFORM_CAM0_TO_CAM1 = np.array(
    [
        [0.99997614, 0.00469811, -0.00506377, 0.10817339],
        [-0.00470555, 0.99998786, -0.00145908, 0.00051284],
        [0.00505686, 0.00148287, 0.99998611, 0.00069196],
        [0, 0, 0, 1],
    ]
)

HILTI_IMAGE_SIZE = (720, 540)
