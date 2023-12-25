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

from prime_slam.observation.observation_creator import ObservationCreator
from prime_slam.slam.mapping.mapping import Mapping
from prime_slam.slam.slam_config import SLAMConfig
from prime_slam.slam.tracking.tracker import Tracker

__all__ = ["SLAMModuleFactory"]


class SLAMModuleFactory:
    def __init__(self, configs: List[SLAMConfig]):
        self._configs = configs

    def create_tracker(self) -> Tracker:
        return Tracker([slam_config.tracking_config for slam_config in self._configs])

    def create_mapping(self) -> Mapping:
        return Mapping([slam_config.mapping_config for slam_config in self._configs])

    def create_observation_creator(self) -> ObservationCreator:
        return ObservationCreator(
            [slam_config.observation_config for slam_config in self._configs]
        )
