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

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.frame.keyframe_selection.keyframe_selector import KeyframeSelector

__all__ = ["StatisticalKeyframeSelector"]


class StatisticalKeyframeSelector(KeyframeSelector):
    def __init__(
        self,
        min_step: int = 5,
        tracked_points_ratio_threshold: float = 0.4,
        min_tracked_points_number: int = 20,
    ):
        # TODO: make it adaptive
        self.min_step = min_step
        self.tracked_points_ratio_threshold = tracked_points_ratio_threshold
        self.min_tracked_points_number = min_tracked_points_number
        self.step_counter = 0

    def is_selected(self, keyframe: Frame, data) -> bool:
        self.step_counter += 1
        if self.step_counter < self.min_step:
            return False

        selected = True
        for name in data.names:
            tracked_points_number = len(data.get_matched_target(name))
            untracked_points_number = len(data.get_unmatched_target(name))
            ratio = tracked_points_number / (
                tracked_points_number + untracked_points_number + 1
            )
            selected &= (ratio < self.tracked_points_ratio_threshold) and (
                tracked_points_number > self.min_tracked_points_number
            )

        if selected:
            self.step_counter = 0

        return selected
