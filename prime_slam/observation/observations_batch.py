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

from typing import Dict, List

from prime_slam.observation.keyobject import Keyobject
from prime_slam.observation.observation import Observation
from prime_slam.sensor.rgbd import RGBDImage
from prime_slam.typing.hints import ArrayNxM

__all__ = ["ObservationsBatch", "ObservationData"]


class ObservationData:
    def __init__(
        self,
        observations: List[Observation],
        observation_name: str,
        sensor_measurement: RGBDImage,
    ):
        self._name = observation_name
        self._sensor_measurement = sensor_measurement

        self._descriptors = np.array(
            [observation.descriptor for observation in observations]
        )
        self._coordinates = np.array(
            [observation.keyobject.coordinates for observation in observations]
        )
        self._uncertainties = np.array(
            [observation.keyobject.uncertainty for observation in observations]
        )
        self._keyobjects = [observation.keyobject for observation in observations]

    @property
    def name(self):
        return self._name

    @property
    def sensor_measurement(self):
        return self._sensor_measurement

    @property
    def descriptors(self) -> ArrayNxM[float]:
        return self._descriptors

    @property
    def coordinates(self) -> ArrayNxM[float]:
        return self._coordinates

    @property
    def uncertainties(self) -> ArrayNxM[float]:
        return self._uncertainties

    @property
    def keyobjects(self) -> List[Keyobject]:
        return self._keyobjects

    def __len__(self):
        return len(self.keyobjects)


class ObservationsBatch:
    def __init__(
        self,
        observations_batch: Dict[str, ObservationData] = None,
    ):
        self._observations_batch = (
            observations_batch if observations_batch is not None else {}
        )

    def __getitem__(self, index):
        return self._observations_batch[index]

    def __setitem__(self, index, value):
        self._observations_batch[index] = value

    def get_observation_data(self, observation_name: str) -> ObservationData:
        return self._observations_batch[observation_name]

    @property
    def observation_names(self) -> List[str]:
        return list(self._observations_batch.keys())
