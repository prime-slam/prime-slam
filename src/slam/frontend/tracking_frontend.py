import numpy as np

from typing import List

from src.tracking.data_association import DataAssociation
from src.frame import Frame
from src.geometry.pose import Pose
from src.graph.factor_graph import FactorGraph
from src.frame.keyframe_selection.keyframe_selector import KeyframeSelector
from src.mapping.mapping import Mapping
from src.observation.keyobject import Keyobject
from src.observation.observation_creator import ObservationsCreator
from src.observation.observations_batch import ObservationsBatch
from src.sensor.sensor_data import SensorData
from src.slam.frontend.frontend import Frontend
from src.tracking.tracker import Tracker
from src.utils.context_counter import ContextCounter

__all__ = ["TrackingFrontend"]


class TrackingFrontend(Frontend):
    def __init__(
        self,
        observation_creators: List[ObservationsCreator],
        keyframe_selector: KeyframeSelector,
        initial_pose,
    ):
        self.observation_creators = observation_creators
        self.initial_pose: Pose = initial_pose
        self._graph = FactorGraph()
        self.keyframe_counter = ContextCounter()
        self.tracker = Tracker(observation_creators)
        self.mapping = Mapping(observation_creators)
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
            observation_batch = []
            names = []
            for observation_creator in self.observation_creators:
                observations = observation_creator.create_observations(sensor_data)
                observation_batch.append(observations)
                names.append(observation_creator.observation_name)

            observations_batch = ObservationsBatch(observation_batch, names)

            frame = Frame(
                observations_batch,
                sensor_data,
                identifier=current_id,
            )

        if frame.identifier == 0:
            self.__initialize(frame)
        else:
            tracking_result = self.__track(frame)
            frame.update_pose(tracking_result.pose)
            local_map = self.mapping.create_local_map(frame)
            frame.local_map = local_map

            if self.keyframe_selector.is_selected(frame):
                frame.is_keyframe = True
                self.__insert_new_keyframe(frame, tracking_result.associations)
        return frame

    def __insert_new_keyframe(self, new_frame: Frame, map_association: DataAssociation):
        self.keyframes.append(new_frame)
        self._graph.add_pose_node(
            node_id=new_frame.identifier,
            pose=new_frame.world_to_camera_transform,
        )
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name

            new_keypoints_index = map_association.get_matched_reference(
                observation_name
            )
            map_keypoints_index = map_association.get_matched_target(observation_name)
            keyobjects: List[Keyobject] = new_frame.observations.get_keyobjects(
                observation_name
            )
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
                self._graph.add_landmark_node(landmark_id, landmark_position)
                self._graph.add_observation_factor(
                    pose_id=new_frame.identifier,
                    landmark_id=landmark_id,
                    observation=keyobjects[unmatched_index].coordinates,
                    sensor_measurement=new_frame.sensor_measurement,
                    information=keyobjects[unmatched_index].uncertainty,
                )
                self.mapping.add_landmark(landmark, observation_name)

    def __track(self, new_frame: Frame):
        prev_frame = self.keyframes[-1]
        relative_tracking_result = self.tracker.track(prev_frame, new_frame)
        new_frame.update_pose(relative_tracking_result.pose)
        visible_map = self.mapping.get_visible_multimap(new_frame)
        map_tracking_result = self.tracker.track_map(new_frame, visible_map)

        return map_tracking_result

    def __initialize(self, keyframe: Frame):
        keyframe.update_pose(self.initial_pose)
        self.keyframes.append(keyframe)
        self.mapping.initialize_map(keyframe)
        self._graph.add_pose_node(
            node_id=keyframe.identifier, pose=keyframe.world_to_camera_transform
        )
        for observation_name in keyframe.local_map.landmark_names:
            keyobjects: List[Keyobject] = keyframe.observations.get_keyobjects(
                observation_name
            )
            landmarks = keyframe.local_map.get_landmarks(observation_name)
            for landmark, keyobject in zip(landmarks, keyobjects):
                self._graph.add_landmark_node(landmark.identifier, landmark.position)
                self._graph.add_observation_factor(
                    pose_id=keyframe.identifier,
                    landmark_id=landmark.identifier,
                    observation=keyobject.coordinates,
                    sensor_measurement=keyframe.sensor_measurement,
                    information=keyobject.uncertainty,
                )

    def update_poses(self, new_poses):
        self._graph.update_poses(new_poses)
        for kf, new_pose in zip(self.keyframes, new_poses):
            kf.update_pose(Pose(new_pose))

    def update_landmark_positions(self, new_positions, landmark_name):
        self._graph.update_landmarks_positions(new_positions)
        self.mapping.multi_map.update_position(
            new_positions, landmark_name
        )  # TODO: add methods to mapping
        self.mapping.multi_map.recalculate_mean_viewing_directions(landmark_name)
