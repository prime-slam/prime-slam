from typing import List

import numpy as np

from src.data_association import DataAssociation
from src.frame import Frame
from src.graph.factor_graph import FactorGraph
from src.keyframe_selection.keyframe_selector import KeyframeSelector
from src.mapping.mapping import Mapping

from src.observation.keyobject import Keyobject
from src.observation.observation_creator import ObservationsCreator
from src.observation.observations_batch import ObservationsBatch
from src.sensor.sensor_data_base import SensorData
from src.slam.frontend import Frontend


class TrackingFrontend(Frontend):
    # TODO: make counter with keyword `with`
    def __init__(
        self,
        observation_creators: List[ObservationsCreator],
        keyframe_selector: KeyframeSelector,
        initial_pose,
    ):
        self.observation_creators = observation_creators
        self.initial_pose = initial_pose
        self.graph = FactorGraph()
        self.keyframe_counter = 0
        self.mapping = Mapping(observation_creators)
        self.keyframe_selector = keyframe_selector

    def initialize_tracking(self, keyframe: Frame):
        keyframe.update_pose(self.initial_pose)
        self.mapping.initialize_map(keyframe)
        self.graph.add_pose_node(
            node_id=keyframe.identifier, pose=keyframe.world_to_camera_transform
        )
        for observation_name in keyframe.local_map.landmark_names:
            keyobjects: List[Keyobject] = keyframe.observations.get_keyobjects(
                observation_name
            )
            landmarks = keyframe.local_map.get_landmarks(observation_name)
            for landmark, keyobject in zip(landmarks, keyobjects):
                self.graph.add_landmark_node(landmark.identifier, landmark.position)
                self.graph.add_observation_factor(
                    pose_id=keyframe.identifier,
                    landmark_id=landmark.identifier,
                    observation=keyobject.coordinates,
                    sensor_measurement=keyframe.sensor_measurement,
                    information=keyobject.uncertainty,
                )

    def process_sensor_data(self, sensor_data: SensorData) -> Frame:
        keyobjects_batch = []
        descriptors_batch = []
        names = []
        for observation_creator in self.observation_creators:
            keyobjects, decriptors = observation_creator.create_observations(
                sensor_data
            )
            keyobjects_batch.append(keyobjects)
            descriptors_batch.append(decriptors)
            names.append(observation_creator.observation_name)

        observations_batch = ObservationsBatch(
            keyobjects_batch, descriptors_batch, names
        )

        frame = Frame(
            observations_batch,
            sensor_data,
            identifier=self.keyframe_counter,
        )
        self.keyframe_counter += 1
        return frame

    def track(
        self,
        prev_frame: Frame,
        new_frame: Frame,
        sensor_data: SensorData,
    ):
        data_association = DataAssociation()
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name
            initial_matches = observation_creator.matcher(
                new_frame.observations.get_descriptors(observation_name),
                prev_frame.observations.get_descriptors(observation_name),
            )
            initial_relative_pose = (
                observation_creator.pose_estimator.estimate_relative_pose(
                    new_frame,
                    prev_frame,
                    initial_matches,
                    observation_name,
                )
            )
            initial_absolute_pose = (
                initial_relative_pose @ prev_frame.world_to_camera_transform
            )
            new_frame.update_pose(initial_absolute_pose)

            current_map = self.mapping.map
            landmarks = current_map.get_landmarks(observation_name)
            landmark_positions = current_map.get_positions(observation_name)
            landmark_positions_cam = observation_creator.projector.transform(
                landmark_positions, initial_absolute_pose
            )
            map_mean_viewing_directions = current_map.get_mean_viewing_directions(
                observation_name
            )
            projected_map = observation_creator.projector.project(
                landmark_positions_cam,
                sensor_data.depth.intrinsics,
                np.eye(4),
            )
            # TODO: move from this filter
            height, width = sensor_data.depth.depth_map.shape[:2]
            depth_mask = landmark_positions_cam[:, 2] > 0
            origin = new_frame.origin
            viewing_directions = landmark_positions - origin
            viewing_directions = viewing_directions / np.linalg.norm(
                viewing_directions, axis=-1
            ).reshape(-1, 1)

            viewing_direction_mask = (
                np.sum(map_mean_viewing_directions * viewing_directions, axis=-1) >= 0.5
            )
            mask = (
                (projected_map[:, 0] >= 0)
                & (projected_map[:, 0] < width)
                & (projected_map[:, 1] >= 0)
                & (projected_map[:, 1] < height)
                & depth_mask
                # & viewing_direction_mask
            )

            map_indices_masked = np.where(mask)[0]
            map_descriptors_masked = current_map.get_descriptors(observation_name)[mask]
            matches = observation_creator.matcher(
                new_frame.observations.get_descriptors(observation_name),
                map_descriptors_masked,
            )
            matches[:, 1] = map_indices_masked[matches[:, 1]]

            unmatched_reference_indices = np.setdiff1d(
                np.arange(
                    len(new_frame.observations.get_descriptors(observation_name))
                ),
                matches[:, 0],
            )
            data_association.set_associations(
                observation_name,
                reference_indices=matches[:, 0],
                target_indices=[landmarks[index].identifier for index in matches[:, 1]],
                unmatched_reference_indices=unmatched_reference_indices,
                unmatched_target_indices=unmatched_reference_indices,
            )
            absolute_pose_delta = (
                observation_creator.pose_estimator.estimate_absolute_pose(
                    new_frame,
                    landmark_positions_cam,
                    matches,
                    observation_name,
                )
            )
            absolute_pose = absolute_pose_delta @ initial_absolute_pose
            new_frame.update_pose(absolute_pose)
        self.mapping.create_local_map(new_frame)

        if self.keyframe_selector.is_selected(new_frame):
            new_frame.is_keyframe = True
            self.add_new_keyframe(new_frame, data_association)
        return new_frame

    def add_new_keyframe(self, new_frame: Frame, data_association: DataAssociation):
        self.graph.add_pose_node(
            node_id=new_frame.identifier,
            pose=new_frame.world_to_camera_transform,
        )
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name

            new_keypoints_index = data_association.get_matched_reference(
                observation_name
            )
            map_keypoints_index = data_association.get_matched_target(observation_name)
            keyobjects: List[Keyobject] = new_frame.observations.get_keyobjects(
                observation_name
            )
            new_landmarks = new_frame.local_map.get_landmarks(observation_name)
            depth_map = new_frame.sensor_measurement.depth.depth_map

            # add matched correspondences
            for new_index, map_index in zip(new_keypoints_index, map_keypoints_index):
                x, y = keyobjects[new_index].coordinates.astype(int)
                # TODO: remove from this
                if depth_map[y, x] == 0:
                    continue
                self.graph.add_observation_factor(
                    pose_id=new_frame.identifier,
                    landmark_id=map_index,
                    observation=keyobjects[new_index].coordinates,
                    sensor_measurement=new_frame.sensor_measurement,
                    information=keyobjects[new_index].uncertainty,
                )
            # self.mapping.add_associations(
            #     observation_name, new_frame, map_keypoints_index
            # )
            unmatched_indices = data_association.get_unmatched_reference(
                observation_name
            )

            for unmatched_index in unmatched_indices:
                landmark = new_landmarks[unmatched_index]
                landmark_id = landmark.identifier
                landmark_position = landmark.position
                if np.logical_or.reduce(np.isnan(landmark_position), axis=-1):
                    continue
                self.graph.add_landmark_node(landmark_id, landmark_position)
                self.graph.add_observation_factor(
                    pose_id=new_frame.identifier,
                    landmark_id=landmark_id,
                    observation=keyobjects[unmatched_index].coordinates,
                    sensor_measurement=new_frame.sensor_measurement,
                    information=keyobjects[unmatched_index].uncertainty,
                )
                self.mapping.add_landmark(landmark, observation_name)
