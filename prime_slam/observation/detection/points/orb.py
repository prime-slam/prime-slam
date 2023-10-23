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

import cv2
import numpy as np

from prime_slam.observation.detection.points.opencv_point_detector import (
    OpenCVPointDetector,
)
from prime_slam.typing.hints import ArrayN

__all__ = ["ORB"]


class ORB(OpenCVPointDetector):
    def __init__(
        self, features_number: int, scale_factor: float = 1.2, levels_number=8
    ):
        super().__init__(
            cv2.ORB_create(features_number, scale_factor, levels_number),
            self.create_squared_sigma_levels(levels_number, scale_factor),
        )

    @staticmethod
    def create_squared_sigma_levels(
        levels_number: int = 8,
        scale_factor: float = 1.2,
        init_sigma_level: float = 1.0,
        init_scale_factor: float = 1.0,
    ) -> ArrayN[float]:
        scale_factors = np.zeros(levels_number)
        squared_sigma_levels = np.zeros(levels_number)

        scale_factors[0] = init_scale_factor
        squared_sigma_levels[0] = init_sigma_level * init_sigma_level

        for i in range(1, levels_number):
            scale_factors[i] = scale_factors[i - 1] * scale_factor
            squared_sigma_levels[i] = (
                scale_factors[i] * scale_factors[i] * squared_sigma_levels[0]
            )

        return squared_sigma_levels
