from typing import List

import numpy as np

from src.graph.factor_graph import FactorGraph
from src.keyframe import Keyframe
from src.keyframe_selection.keyframe_selector import KeyframeSelector
from src.landmark import Landmark
from src.map import Map
from src.observation.keyobject import Keyobject
from src.observation.observation_creator import ObservationsCreator
from src.observation.observations_batch import ObservationsBatch


class PrimeSLAMFrontend:
    def __init__(
        self,
        observation_creators: List[ObservationsCreator],
        keyframe_selector: KeyframeSelector,
        init_pose,
    ):
        self.observation_creators = observation_creators
        self.keyframes: List[Keyframe] = []
        self.keyframe_selector = keyframe_selector
        self.map = Map()
        self.init_pose = init_pose
        self.keyframe_counter = 0
        self.graph = FactorGraph()

    def initialize_map(self, keyframe: Keyframe):
        keyframe.update_pose(np.linalg.inv(self.init_pose))

        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name
            keyobjects = keyframe.observations.get_keyobjects(observation_name)
            landmark_positions = observation_creator.projector.back_project(
                np.array([keyobject.coordinates for keyobject in keyobjects]),
                keyframe.sensor_measurement.depth.depth_map,
                keyframe.sensor_measurement.depth.depth_scale,
                keyframe.sensor_measurement.depth.intrinsics,
                self.init_pose,
            )
            descriptors = keyframe.observations.get_descriptors(observation_name)
            origin = keyframe.origin

            for landmark_id, (keyobject, landmark_position, descriptor) in enumerate(
                zip(keyobjects, landmark_positions, descriptors)
            ):
                viewing_direction = landmark_position - origin
                self.graph.add_landmark_node(landmark_id, landmark_position)
                self.graph.add_observation_factor(
                    pose_id=keyframe.identifier,
                    landmark_id=landmark_id,
                    observation=keyobject.coordinates,
                    sensor_measurement=keyframe.sensor_measurement,
                    information=keyobject.uncertainty,
                )
                self.map.add_landmark(
                    Landmark(landmark_position, descriptor, viewing_direction),
                    observation_name,
                )

        self.graph.add_pose_node(
            node_id=keyframe.identifier, pose=keyframe.world_to_camera_transform
        )

        self.keyframes.append(keyframe)
        self.keyframe_counter += 1

    def process_frame(self, sensor_data):
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

        new_keyframe = Keyframe(observations_batch, sensor_data)
        if len(self.keyframes) == 0:
            self.initialize_map(new_keyframe)
            return

        last_keyframe = self.keyframes[-1]
        matches_batch = {}
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name
            initial_matches = observation_creator.matcher(
                observations_batch.get_descriptors(observation_name),
                last_keyframe.observations.get_descriptors(observation_name),
            )
            initial_relative_pose = (
                observation_creator.pose_estimator.estimate_relative_pose(
                    new_keyframe,
                    last_keyframe,
                    initial_matches,
                    observation_name,
                )
            )
            initial_absolute_pose = (
                initial_relative_pose @ last_keyframe.world_to_camera_transform
            )
            new_keyframe.update_pose(initial_absolute_pose)

            map_positions = self.map.get_positions(observation_name)
            map_positions_cam = observation_creator.projector.transform(
                self.map.get_positions(observation_name), initial_absolute_pose
            )
            map_mean_viewing_directions = self.map.get_mean_viewing_directions(
                observation_name
            )
            projected_map = observation_creator.projector.project(
                map_positions_cam,
                sensor_data.depth.intrinsics,
                np.eye(4),
            )

            height, width = sensor_data.depth.depth_map.shape[:2]
            depth_mask = map_positions_cam[:, 2] > 0
            origin = new_keyframe.origin
            viewing_directions = map_positions - origin
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
                & viewing_direction_mask
            )

            map_indices_masked = np.where(mask)[0]
            map_descriptors_masked = self.map.get_descriptors(observation_name)[mask]
            matches = observation_creator.matcher(
                observations_batch.get_descriptors(observation_name),
                map_descriptors_masked,
            )
            matches[:, 1] = map_indices_masked[matches[:, 1]]
            matches_batch[observation_name] = matches
            absolute_pose_delta = (
                observation_creator.pose_estimator.estimate_absolute_pose(
                    new_keyframe,
                    map_positions_cam,
                    matches,
                    observation_name,
                )
            )
            absolute_pose = absolute_pose_delta @ initial_absolute_pose
            new_keyframe.update_pose(absolute_pose)

        if self.keyframe_selector.is_selected(new_keyframe):
            new_keyframe.identifier = self.keyframe_counter
            origin = new_keyframe.origin
            self.graph.add_pose_node(
                node_id=new_keyframe.identifier,
                pose=new_keyframe.camera_to_world_transform,
            )
            for observation_creator in self.observation_creators:
                observation_name = observation_creator.observation_name
                matches = matches_batch[observation_name]

                new_keypoints_index = matches[:, 0]
                map_keypoints_index = matches[:, 1]
                keyobjects: List[Keyobject] = new_keyframe.observations.get_keyobjects(
                    observation_name
                )
                landmark_positions = observation_creator.projector.back_project(
                    np.array([keyobject.coordinates for keyobject in keyobjects]),
                    new_keyframe.sensor_measurement.depth.depth_map,
                    new_keyframe.sensor_measurement.depth.depth_scale,
                    new_keyframe.sensor_measurement.depth.intrinsics,
                    self.init_pose,
                )

                descriptors = new_keyframe.observations.get_descriptors(
                    observation_name
                )

                # add matched correspondances
                for new_index, map_index in zip(
                    new_keypoints_index, map_keypoints_index
                ):
                    landmark_position = self.map.get_landmarks(observation_name)[
                        map_index
                    ].position
                    self.graph.add_observation_factor(
                        pose_id=new_keyframe.identifier,
                        landmark_id=map_index,
                        observation=keyobjects[new_index].coordinates,
                        sensor_measurement=new_keyframe.sensor_measurement,
                        information=keyobjects[new_index].uncertainty,
                    )
                    self.map.add_viewing_direction(
                        observation_name, map_index, landmark_position - origin
                    )
                unmatched_indices = np.setdiff1d(
                    np.arange(
                        len(observations_batch.get_descriptors(observation_name))
                    ),
                    new_keypoints_index,
                )

                for unmatched_index in unmatched_indices:
                    landmark_id = self.map.get_size(observation_name)
                    landmark_position = landmark_positions[unmatched_index]
                    viewing_direction = landmark_position - origin
                    self.graph.add_landmark_node(landmark_id, landmark_position)
                    self.graph.add_observation_factor(
                        pose_id=new_keyframe.identifier,
                        landmark_id=landmark_id,
                        observation=keyobjects[unmatched_index].coordinates,
                        sensor_measurement=new_keyframe.sensor_measurement,
                        information=keyobjects[unmatched_index].uncertainty,
                    )
                    self.map.add_landmark(
                        Landmark(
                            landmark_position,
                            descriptors[unmatched_index],
                            viewing_direction,
                        ),
                        observation_name,
                    )

            self.keyframes.append(new_keyframe)
            self.keyframe_counter += 1
