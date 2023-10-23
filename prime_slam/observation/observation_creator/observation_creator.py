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

from typing import List

from prime_slam.observation.observation import Observation
from prime_slam.observation.observations_batch import ObservationsBatch
from prime_slam.sensor.sensor_data import SensorData
from prime_slam.observation.observation_creator.observation_config import (
    ObservationConfig,
)


__all__ = ["ObservationCreator"]


class ObservationCreator:
    def __init__(self, observation_configs: List[ObservationConfig]):
        self.observation_configs = observation_configs

    def create_observations(self, sensor_data: SensorData) -> ObservationsBatch:
        observation_batch = []
        names = []
        for config in self.observation_configs:
            keyobjects = config.detector.detect(sensor_data)
            descriptors = config.descriptor.descript(keyobjects, sensor_data)
            observations = [
                Observation(keyobject, descriptor)
                for keyobject, descriptor in zip(keyobjects, descriptors)
            ]

            observations = config.observations_filter.apply(observations, sensor_data)
            observation_batch.append(observations)
            names.append(config.observation_name)

        observations_batch = ObservationsBatch(observation_batch, names)

        return observations_batch
