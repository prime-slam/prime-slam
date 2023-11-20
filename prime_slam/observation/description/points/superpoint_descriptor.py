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

from pathlib import Path
from typing import List

from external.superpoint.demo_superpoint import SuperPointFrontend
from prime_slam.observation.description.descriptor import Descriptor
from prime_slam.observation.keyobject import Keyobject
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["SuperPointDescriptor"]

from prime_slam.typing.hints import ArrayNxM


class SuperPointDescriptor(Descriptor):
    def __init__(
        self, weights_path: Path = Path("external/superpoint/superpoint_v1.pth")
    ):
        self.model = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=4,
            conf_thresh=0.015,
            nn_thresh=0.7,
            cuda=True,
        )

    def descript(
        self,
        observations: List[Keyobject],
        sensor_data: SensorData,
    ) -> ArrayNxM[float]:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        scaled_gray = gray.astype(np.float32) / 255.0
        _, desc, _ = self.model.run(scaled_gray)
        return desc.T
