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

from prime_slam.slam.slam_module_factory import SLAMModuleFactory
from prime_slam.slam.tracking.data_association import DataAssociation
from prime_slam.slam.frame.frame import Frame
from prime_slam.geometry.pose import Pose
from prime_slam.slam.graph.factor_graph import FactorGraph
from prime_slam.slam.frame.keyframe_selection.keyframe_selector import KeyframeSelector
from prime_slam.observation.keyobject import Keyobject
from prime_slam.sensor.sensor_data import SensorData
from prime_slam.slam.frontend.frontend import Frontend
from prime_slam.utils.context_counter import ContextCounter

__all__ = ["TrackingFrontend"]


class TrackingFrontend(Frontend):
    def __init__(
        self,
        module_factory: SLAMModuleFactory,
        keyframe_selector: KeyframeSelector,
        initial_pose: Pose,
    ):
        self.initial_pose = initial_pose
        self._graph = FactorGraph()
        self.keyframe_counter = ContextCounter()
        self.observation_creator = module_factory.create_observation_creator()
        self.tracker = module_factory.create_tracker()
        self.mapping = module_factory.create_mapping()
        self.keyframe_selector = keyframe_selector
        self.keyframes: List[Frame] = []

    @property
    def graph(self):
        return self._graph

    @property
    def map(self):
        return self.mapping.multi_map

    @property
    def trajectory(self):
        return [kf.world_to_camera_transform for kf in self.keyframes]

    def process_sensor_data(self, sensor_data: SensorData) -> Frame:
        with self.keyframe_counter as current_id:
            observations_batch = self.observation_creator.create_observations(
                sensor_data
            )
            frame = Frame(
                observations_batch,
                sensor_data,
                identifier=current_id,
            )

        if frame.identifier == 0:
            self.__initialize(frame)
        else:
            relative_tracking_result, map_tracking_result = self.__track(frame)
            frame.update_pose(map_tracking_result.pose)
            local_map = self.mapping.create_local_map(frame)
            frame.local_map = local_map

            if self.keyframe_selector.is_selected(
                frame, relative_tracking_result.associations
            ):
                frame.is_keyframe = True
                self.__insert_new_keyframe(frame, map_tracking_result.associations)
        return frame

    def update_graph(self):
        self._graph.update()

    def update_poses(self, new_poses):
        for kf, new_pose in zip(self.keyframes, new_poses):
            kf.update_pose(Pose(new_pose))

    def update_landmark_positions(self, new_positions, landmark_name):
        self.mapping.update_landmark_positions(new_positions, landmark_name)

    def __insert_new_keyframe(self, new_frame: Frame, map_association: DataAssociation):
        self.keyframes.append(new_frame)
        self._graph.add_pose_node(new_frame)
        for observation_name in new_frame.observations.observation_names:
            new_keypoints_index = map_association.get_matched_reference(
                observation_name
            )
            map_keypoints_index = map_association.get_matched_target(observation_name)
            keyobjects = new_frame.observations.get_keyobjects(observation_name)
            new_landmarks = new_frame.local_map.get_landmarks(observation_name)

            # add matched correspondences
            for new_index, map_index in zip(new_keypoints_index, map_keypoints_index):
                self._graph.add_observation_factor(
                    pose_id=new_frame.identifier,
                    landmark_id=map_index,
                    observation=keyobjects[new_index].coordinates,
                    sensor_measurement=new_frame.sensor_measurement,
                    information=keyobjects[new_index].uncertainty,
                )
            self.mapping.add_associations(
                observation_name, new_frame, map_keypoints_index
            )
            unmatched_indices = map_association.get_unmatched_reference(
                observation_name
            )

            for unmatched_index in unmatched_indices:
                landmark = new_landmarks[unmatched_index]
                landmark_id = landmark.identifier
                landmark_position = landmark.position
                if np.logical_or.reduce(np.isnan(landmark_position), axis=-1):
                    continue
                self._graph.add_landmark_node(landmark)
                self._graph.add_observation_factor(
                    pose_id=new_frame.identifier,
                    landmark_id=landmark_id,
                    observation=keyobjects[unmatched_index].coordinates,
                    sensor_measurement=new_frame.sensor_measurement,
                    information=keyobjects[unmatched_index].uncertainty,
                )
                self.mapping.add_landmark(landmark, observation_name)

        self.mapping.cull_landmarks()
        self.update_graph()

    def __track(self, new_frame: Frame):
        prev_frame = self.keyframes[-1]
        relative_tracking_result = self.tracker.track(prev_frame, new_frame)
        new_frame.update_pose(relative_tracking_result.pose)
        visible_map = self.mapping.get_visible_multimap(new_frame)
        map_tracking_result = self.tracker.track_map(new_frame, visible_map)

        return relative_tracking_result, map_tracking_result

    def __initialize(self, keyframe: Frame):
        keyframe.update_pose(self.initial_pose)
        self.keyframes.append(keyframe)
        self.mapping.initialize_map(keyframe)
        self._graph.add_pose_node(keyframe)
        for observation_name in keyframe.local_map.landmark_names:
            keyobjects: List[Keyobject] = keyframe.observations.get_keyobjects(
                observation_name
            )
            landmarks = keyframe.local_map.get_landmarks(observation_name)
            for landmark, keyobject in zip(landmarks, keyobjects):
                self._graph.add_landmark_node(landmark)
                self._graph.add_observation_factor(
                    pose_id=keyframe.identifier,
                    landmark_id=landmark.identifier,
                    observation=keyobject.coordinates,
                    sensor_measurement=keyframe.sensor_measurement,
                    information=keyobject.uncertainty,
                )
