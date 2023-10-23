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

__all__ = ["EveryNthKeyframeSelector"]


class EveryNthKeyframeSelector(KeyframeSelector):
    def __init__(self, n: int):
        self.n = n
        self.counter = 0

    def is_selected(self, keyframe: Frame) -> bool:
        self.counter += 1
        selected = (self.counter // self.n) == 1
        if selected:
            self.counter = 0

        return selected
