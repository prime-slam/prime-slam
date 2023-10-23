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
from prime_slam.observation.keyline import Keyline
from prime_slam.observation.keyobject import Keyobject
from prime_slam.sensor.rgbd import RGBDImage

__all__ = ["LSD"]


class LSD(Detector):
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

    def detect(self, sensor_data: RGBDImage) -> List[Keyobject]:
        opencv_image = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)

        lines, _, _, scores = self.lsd.detect(opencv_image)
        lines = lines.flatten().reshape(-1, 4)
        observations = [
            Keyline(x1, y1, x2, y2, uncertainty=score)
            for (x1, y1, x2, y2), score in zip(lines, scores)
        ]

        return observations
