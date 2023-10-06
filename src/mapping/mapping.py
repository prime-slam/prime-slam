from typing import List

import numpy as np

from src.frame import Frame
from src.mapping.landmark import Landmark
from src.mapping.map import Map
from src.observation.keyobject import Keyobject
from src.observation.observation_creator import ObservationsCreator
from src.utils.context_counter import ContextCounter


class Mapping:
    def __init__(self, observation_creators: List[ObservationsCreator]):
        self.landmarks_counter = ContextCounter()
        self.observation_creators = observation_creators
        self.map = Map()

    def create_local_map(self, frame: Frame):
        local_map = Map()
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name
            keyobjects: List[Keyobject] = frame.observations.get_keyobjects(
                observation_name
            )
            landmark_positions = observation_creator.projector.back_project(
                np.array([keyobject.coordinates for keyobject in keyobjects]),
                frame.sensor_measurement.depth.depth_map,
                frame.sensor_measurement.depth.depth_scale,
                frame.sensor_measurement.depth.intrinsics,
                frame.world_to_camera_transform,
            )
            descriptors = frame.observations.get_descriptors(observation_name)
            for landmark_position, descriptor in zip(landmark_positions, descriptors):
                with self.landmarks_counter as current_id:
                    local_map.add_landmark(
                        Landmark(
                            current_id,
                            landmark_position,
                            descriptor,
                            frame,
                        ),
                        observation_name,
                    )
        frame.local_map = local_map

    def initialize_map(self, frame):
        self.create_local_map(frame)
        self.map = frame.local_map

    def add_associations(self, observation_name, frame, landmark_ids):
        for landmark_id in landmark_ids:
            self.map.add_associated_keyframe(observation_name, landmark_id, frame)

    def add_landmark(self, landmark, observation_name):
        self.map.add_landmark(
            landmark,
            observation_name,
        )
