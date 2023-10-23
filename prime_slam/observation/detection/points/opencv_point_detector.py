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

from typing import List

from prime_slam.observation.detection.detector import Detector
from prime_slam.observation.keyobject import Keyobject
from prime_slam.observation.opencv_keypoint import OpenCVKeypoint
from prime_slam.sensor.rgbd import RGBDImage
from prime_slam.typing.hints import ArrayN

__all__ = ["OpenCVPointDetector"]


class OpenCVPointDetector(Detector):
    def __init__(self, detector, squared_sigma_levels: ArrayN[float]):
        self.detector = detector
        self.squared_sigma_levels = squared_sigma_levels
        self.inv_squared_sigma_levels = 1 / squared_sigma_levels

    def detect(self, sensor_data: RGBDImage) -> List[Keyobject]:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        keypoints = self.detector.detect(gray, None)

        return [
            OpenCVKeypoint(kp, self.inv_squared_sigma_levels[kp.octave])
            for kp in keypoints
        ]
