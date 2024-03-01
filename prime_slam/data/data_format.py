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

from enum import Enum

__all__ = ["DataFormatRGBD", "DataFormatStereo"]


class DataFormatRGBD(Enum):
    tum = 0
    icl = 1
    icl_tum = 2

    @staticmethod
    def to_string(delimiter: str = ", "):
        return delimiter.join(dist.name for dist in DataFormatRGBD)


class DataFormatStereo(Enum):
    stereo = 0
    stereo_lidar = 1

    @staticmethod
    def to_string(delimiter: str = ", "):
        return delimiter.join(dist.name for dist in DataFormatStereo)
