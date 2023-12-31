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

__all__ = ["ContextCounter"]


class ContextCounter:
    def __init__(self, initial_value: int = 0):
        self._value = initial_value

    def __enter__(self) -> int:
        return self._value

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._value += 1

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_value) -> None:
        self._value = new_value
