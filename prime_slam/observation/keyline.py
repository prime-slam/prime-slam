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

from typing import Any

from prime_slam.observation.keyobject import Keyobject
from prime_slam.typing.hints import ArrayN

__all__ = ["Keyline"]


class Keyline(Keyobject):
    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, uncertainty: float = None
    ):
        self._coordinates = np.array([x1, y1, x2, y2])
        self._uncertainty = uncertainty

    @property
    def coordinates(self) -> ArrayN[float]:
        return self._coordinates

    @property
    def uncertainty(self) -> float:
        return self._uncertainty

    @property
    def data(self) -> Any:
        return self._coordinates
