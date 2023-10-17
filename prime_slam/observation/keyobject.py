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

from abc import ABC, abstractmethod
from typing import Any

from prime_slam.typing.hints import ArrayN

__all__ = ["Keyobject"]


class Keyobject(ABC):
    @property
    @abstractmethod
    def coordinates(self) -> ArrayN[float]:
        pass

    @property
    @abstractmethod
    def uncertainty(self) -> float:
        pass

    @property
    @abstractmethod
    def data(self) -> Any:
        pass
