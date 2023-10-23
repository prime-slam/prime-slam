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

__all__ = ["DataAssociation"]


class DataAssociation:
    def __init__(self):
        self._reference_indices = {}
        self._target_indices = {}
        self._unmatched_reference_indices = {}
        self._unmatched_target_indices = {}

    def set_associations(
        self,
        observation_name,
        reference_indices,
        target_indices,
        unmatched_reference_indices,
        unmatched_target_indices,
    ):
        self._reference_indices[observation_name] = reference_indices
        self._target_indices[observation_name] = target_indices
        self._unmatched_reference_indices[
            observation_name
        ] = unmatched_reference_indices
        self._unmatched_target_indices[observation_name] = unmatched_target_indices

    def get_matched_reference(self, observation_name):
        return self._reference_indices[observation_name]

    def get_matched_target(self, observation_name):
        return self._target_indices[observation_name]

    def get_unmatched_reference(self, observation_name):
        return self._unmatched_reference_indices[observation_name]

    def get_unmatched_target(self, observation_name):
        return self._unmatched_target_indices[observation_name]
