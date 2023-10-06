import numpy as np

from typing import List

from src.data_association import DataAssociation
from src.frame import Frame
from src.mapping.map import Map
from src.observation.observation_creator import ObservationsCreator
from src.tracking.tracking_result import TrackingResult


class Tracker:
    def __init__(self, observation_creators: List[ObservationsCreator]):
        self.observation_creators = observation_creators

    def track_map(self, frame: Frame, landmarks_map: Map):
        initial_absolute_pose = frame.world_to_camera_transform
        data_association = DataAssociation()
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name
            landmarks = landmarks_map.get_landmarks(observation_name)
            landmark_positions = landmarks_map.get_positions(observation_name)
            landmark_positions_cam = observation_creator.projector.transform(
                landmark_positions, initial_absolute_pose
            )
            _, mask = observation_creator.projector.get_visible_map_points(
                landmarks_map, frame, observation_name
            )

            map_indices_masked = np.where(mask)[0]
            map_descriptors_masked = landmarks_map.get_descriptors(observation_name)[
                mask
            ]
            matches = observation_creator.matcher(
                frame.observations.get_descriptors(observation_name),
                map_descriptors_masked,
            )
            matches[:, 1] = map_indices_masked[matches[:, 1]]
            absolute_pose_delta = (
                observation_creator.pose_estimator.estimate_absolute_pose(
                    frame,
                    landmark_positions_cam,
                    matches,
                    observation_name,
                )
            )
            reference_indices = matches[:, 0]
            target_indices = [landmarks[index].identifier for index in matches[:, 1]]
            unmatched_reference_indices = np.setdiff1d(
                np.arange(frame.observations.get_size(observation_name)),
                reference_indices,
            )
            unmatched_target_indices = np.setdiff1d(
                landmarks_map.get_size(observation_name),
                target_indices,
            )
            data_association.set_associations(
                observation_name,
                reference_indices=reference_indices,
                target_indices=target_indices,
                unmatched_reference_indices=unmatched_reference_indices,
                unmatched_target_indices=unmatched_target_indices,
            )

            absolute_pose = absolute_pose_delta @ initial_absolute_pose
        return TrackingResult(absolute_pose, data_association)

    def track(
        self,
        prev_frame: Frame,
        new_frame: Frame,
    ):
        data_association = DataAssociation()
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name
            matches = observation_creator.matcher(
                new_frame.observations.get_descriptors(observation_name),
                prev_frame.observations.get_descriptors(observation_name),
            )
            initial_relative_pose = (
                observation_creator.pose_estimator.estimate_relative_pose(
                    new_frame,
                    prev_frame,
                    matches,
                    observation_name,
                )
            )
            initial_absolute_pose = (
                initial_relative_pose @ prev_frame.world_to_camera_transform
            )

            reference_indices = matches[:, 0]
            target_indices = matches[:, 1]
            unmatched_reference_indices = np.setdiff1d(
                np.arange(new_frame.observations.get_size(observation_name)),
                reference_indices,
            )
            unmatched_target_indices = np.setdiff1d(
                np.arange(prev_frame.observations.get_size(observation_name)),
                target_indices,
            )
            data_association.set_associations(
                observation_name,
                reference_indices=reference_indices,
                target_indices=target_indices,
                unmatched_reference_indices=unmatched_reference_indices,
                unmatched_target_indices=unmatched_target_indices,
            )

        return TrackingResult(initial_absolute_pose, data_association)
