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

from prime_slam.sensor.depth import DepthImage
from prime_slam.sensor.rgb import RGBImage
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["RGBDImage"]


class RGBDImage(SensorData):
    def __init__(self, rgb: RGBImage, depth: DepthImage, bf: float = 400):
        self.rgb = rgb
        self.depth = depth
        self.bf = bf
