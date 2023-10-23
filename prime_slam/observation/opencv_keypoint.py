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

from prime_slam.observation.keyobject import Keyobject
from prime_slam.typing.hints import ArrayN

__all__ = ["OpenCVKeypoint"]


class OpenCVKeypoint(Keyobject):
    def __init__(self, keypoint: cv2.KeyPoint, uncertainty: float):
        self.keypoint = keypoint
        self._uncertainty = uncertainty

    @property
    def data(self) -> cv2.KeyPoint:
        return self.keypoint

    @property
    def coordinates(self) -> ArrayN[float]:
        return np.array(self.keypoint.pt)

    @property
    def uncertainty(self) -> float:
        return self._uncertainty
