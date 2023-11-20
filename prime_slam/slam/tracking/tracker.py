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

from typing import List

from prime_slam.geometry.pose import Pose
from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.multi_map import MultiMap
from prime_slam.slam.tracking.data_association import DataAssociation
from prime_slam.slam.tracking.tracking_config import TrackingConfig
from prime_slam.slam.tracking.tracking_result import TrackingResult

__all__ = ["Tracker"]


class Tracker:
    def __init__(self, tracking_configs: List[TrackingConfig]):
        self.tracking_configs = tracking_configs

    def track_map(self, frame: Frame, landmarks_multimap: MultiMap) -> TrackingResult:
        initial_absolute_pose = frame.world_to_camera_transform
        data_association = DataAssociation()
        for config in self.tracking_configs:
            observation_name = config.observation_name
            observations = frame.observations.get_observation_data(observation_name)
            landmarks_map = landmarks_multimap.get_map(observation_name)
            landmarks = landmarks_map.landmarks
            landmark_positions = landmarks_map.positions
            landmark_positions_cam = config.projector.transform(
                landmark_positions, initial_absolute_pose
            )
            matches = config.map_matcher.match_map(
                landmarks_map,
                observations,
            )
            absolute_pose_delta = config.pose_estimator.estimate_absolute_pose(
                observations,
                landmark_positions_cam,
                matches,
            )
            reference_indices = matches[:, 0]
            target_indices = [landmarks[index].identifier for index in matches[:, 1]]
            unmatched_reference_indices = np.setdiff1d(
                np.arange(len(observations)),
                reference_indices,
            )
            unmatched_target_indices = np.setdiff1d(
                np.arange(len(landmarks_map)),
                target_indices,
            )
            data_association.set_associations(
                observation_name,
                reference_indices=reference_indices,
                target_indices=target_indices,
                unmatched_reference_indices=unmatched_reference_indices,
                unmatched_target_indices=unmatched_target_indices,
            )

            absolute_pose = absolute_pose_delta.transformation @ initial_absolute_pose
        return TrackingResult(Pose(absolute_pose), data_association)

    def track(
        self,
        prev_frame: Frame,
        new_frame: Frame,
    ) -> TrackingResult:
        data_association = DataAssociation()
        for config in self.tracking_configs:
            observation_name = config.observation_name
            prev_observations = prev_frame.observations.get_observation_data(
                observation_name
            )
            new_observations = new_frame.observations.get_observation_data(
                observation_name
            )

            matches = config.frame_matcher.match_observations(
                prev_observations,
                new_observations,
            )
            initial_relative_pose = config.pose_estimator.estimate_relative_pose(
                new_observations,
                prev_observations,
                matches,
            )
            initial_absolute_pose = (
                initial_relative_pose.transformation
                @ prev_frame.world_to_camera_transform
            )

            reference_indices = matches[:, 0]
            target_indices = matches[:, 1]

            unmatched_reference_indices = np.setdiff1d(
                np.arange(len(new_observations)),
                reference_indices,
            )
            unmatched_target_indices = np.setdiff1d(
                np.arange(len(prev_observations)),
                target_indices,
            )

            data_association.set_associations(
                observation_name,
                reference_indices=reference_indices,
                target_indices=target_indices,
                unmatched_reference_indices=unmatched_reference_indices,
                unmatched_target_indices=unmatched_target_indices,
            )

        return TrackingResult(Pose(initial_absolute_pose), data_association)
