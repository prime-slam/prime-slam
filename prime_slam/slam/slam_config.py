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

from collections.abc import Callable
from dataclasses import dataclass

from prime_slam.observation.description.descriptor import Descriptor
from prime_slam.observation.detection.detector import Detector
from prime_slam.observation.filter.observation_filter import ObservationsFilter
from prime_slam.projection.projector import Projector
from prime_slam.slam.mapping.map_creator.map_creator import MapCreator
from prime_slam.slam.mapping.mapping_config import MappingConfig
from prime_slam.observation.observation_creator import ObservationConfig
from prime_slam.slam.tracking.tracking_config import TrackingConfig
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator

__all__ = ["SLAMConfig"]


@dataclass
class SLAMConfig:
    detector: Detector
    descriptor: Descriptor
    matcher: Callable[[np.ndarray, np.ndarray], np.ndarray]
    projector: Projector
    pose_estimator: PoseEstimator
    observations_filter: ObservationsFilter
    map_creator: MapCreator
    observation_name: str

    @property
    def tracking_config(self):
        return TrackingConfig(
            matcher=self.matcher,
            projector=self.projector,
            pose_estimator=self.pose_estimator,
            observation_name=self.observation_name,
        )

    @property
    def mapping_config(self):
        return MappingConfig(
            projector=self.projector,
            map_creator=self.map_creator,
            observation_name=self.observation_name,
        )

    @property
    def observation_config(self):
        return ObservationConfig(
            detector=self.detector,
            descriptor=self.descriptor,
            observations_filter=self.observations_filter,
            observation_name=self.observation_name,
        )
