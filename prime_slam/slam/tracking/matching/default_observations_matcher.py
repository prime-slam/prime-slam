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

from skimage.feature import match_descriptors

from functools import partial

from prime_slam.observation.observations_batch import ObservationData
from prime_slam.slam.mapping.map import Map
from prime_slam.slam.tracking.matching.frame_matcher import ObservationsMatcher
from prime_slam.slam.tracking.matching.map_matcher import MapMatcher

__all__ = ["DefaultMatcher"]


class DefaultMatcher(ObservationsMatcher, MapMatcher):
    def __init__(
        self,
        descriptor_matcher=partial(match_descriptors, metric="hamming", max_ratio=0.8),
    ):
        self.descriptor_matcher = descriptor_matcher

    def match_observations(
        self, prev_observations: ObservationData, new_observations: ObservationData
    ):
        return self.descriptor_matcher(
            new_observations.descriptors,
            prev_observations.descriptors,
        )

    def match_map(self, landmark_map: Map, new_observations: ObservationData):
        landmark_descriptors = landmark_map.descriptors
        return self.descriptor_matcher(
            new_observations.descriptors,
            landmark_descriptors,
        )
