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

__all__ = ["SIFT"]


class SIFT(OpenCVPointDetector):
    def __init__(
        self,
        features_number: int,
        scale_factor: float = 2,
        levels_number=8,
        scales_per_octave=3,
    ):
        super().__init__(
            cv2.SIFT_create(features_number, nOctaveLayers=scales_per_octave),
            self.create_squared_sigma_levels(
                levels_number, scale_factor, scales_per_octave=scales_per_octave
            ),
        )

    @staticmethod
    def create_squared_sigma_levels(
        levels_number: int = 8,
        scale_factor: float = 2,
        init_sigma_level: float = 1.6,
        init_scale_factor: float = 1.0,
        scales_per_octave: int = 3,
    ) -> np.ndarray:
        levels_number = scales_per_octave * levels_number + scales_per_octave
        virtual_scale_factor = np.power(scale_factor, 1.0 / scales_per_octave)

        scale_factors = np.zeros(levels_number)
        squared_sigma_levels = np.zeros(levels_number)

        scale_factors[0] = init_scale_factor
        squared_sigma_levels[0] = init_sigma_level * init_sigma_level

        for i in range(1, levels_number):
            scale_factors[i] = scale_factors[i - 1] * virtual_scale_factor
            squared_sigma_levels[i] = (
                scale_factors[i] * scale_factors[i] * squared_sigma_levels[0]
            )

        return squared_sigma_levels
