import numpy as np

from typing import List

from src.frame import Frame
from src.mapping.multi_map import MultiMap
from src.observation.keyobject import Keyobject
from src.observation.observation_creator import ObservationsCreator
from src.utils.context_counter import ContextCounter

__all__ = ["Mapping"]


class Mapping:
    def __init__(self, observation_creators: List[ObservationsCreator]):
        self.landmarks_counter = ContextCounter()
        self.observation_creators = observation_creators  # TODO: only projectors
        self.multi_map = MultiMap()

    def create_local_map(self, frame: Frame):
        local_multimap = MultiMap()
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
            local_map = observation_creator.map_creator.create(
                observation_creator.projector
            )
            for landmark_position, descriptor in zip(landmark_positions, descriptors):
                with self.landmarks_counter as current_id:
                    local_map.add_landmark(
                        local_map.create_landmark(
                            current_id,
                            landmark_position,
                            descriptor,
                            frame,
                        )
                    )

            local_multimap.add_map(observation_name, local_map)
        return local_multimap

    def initialize_map(self, frame):
        local_map = self.create_local_map(frame)
        frame.local_map = local_map
        self.multi_map = local_map

    def add_associations(self, observation_name, frame, landmark_ids):
        for landmark_id in landmark_ids:
            self.multi_map.add_associated_keyframe(observation_name, landmark_id, frame)

    def add_landmark(self, landmark, observation_name):
        self.multi_map.add_landmark(
            landmark,
            observation_name,
        )

    def get_visible_multimap(
        self,
        frame: Frame,
    ):
        visible_multimap = MultiMap()
        for observation_creator in self.observation_creators:
            observation_name = observation_creator.observation_name
            visible_map = self.multi_map.get_map(observation_name).get_visible_map(
                frame
            )
            visible_multimap.add_map(observation_name, visible_map)

        return visible_multimap
