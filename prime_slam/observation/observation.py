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

from prime_slam.observation.keyobject import Keyobject
from prime_slam.typing.hints import ArrayN

__all__ = ["Observation"]


class Observation:
    def __init__(self, keyobject: Keyobject, descriptor: np.ndarray):
        self._keyobject = keyobject
        self._descriptor = descriptor

    @property
    def descriptor(self) -> ArrayN[float]:
        return self._descriptor

    @property
    def keyobject(self) -> Keyobject:
        return self._keyobject
