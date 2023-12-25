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

from dataclasses import dataclass

from prime_slam.projection.projector import Projector
from prime_slam.slam.tracking.matching.frame_matcher import ObservationsMatcher
from prime_slam.slam.tracking.matching.map_matcher import MapMatcher
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator

__all__ = ["TrackingConfig"]


@dataclass
class TrackingConfig:
    frame_matcher: ObservationsMatcher
    map_matcher: MapMatcher
    projector: Projector
    pose_estimator: PoseEstimator
    observation_name: str
