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

from typing import List

from prime_slam.observation.keyobject import Keyobject
from prime_slam.observation.observation import Observation
from prime_slam.typing.hints import ArrayNxM

__all__ = ["ObservationsBatch"]


class ObservationsBatch:
    def __init__(
        self,
        observations_batch: List[List[Observation]],
        names: List[str],
    ):
        self._observations_batch = {}
        self._descriptors_batch = {}
        self._keyobjects_batch = {}
        self._names = names
        for observations, name in zip(observations_batch, names):
            self._observations_batch[name] = observations
            self._descriptors_batch[name] = np.array(
                [observation.descriptor for observation in observations]
            )
            self._keyobjects_batch[name] = [
                observation.keyobject for observation in observations
            ]

    @property
    def observation_names(self) -> List[str]:
        return self._names

    def get_size(self, name) -> int:
        return len(self._observations_batch[name])

    def get_observations(self, name) -> List[Observation]:
        return self._observations_batch[name]

    def get_descriptors(self, name) -> ArrayNxM[float]:
        return self._descriptors_batch[name]

    def get_keyobjects(self, name) -> List[Keyobject]:
        return self._keyobjects_batch[name]
