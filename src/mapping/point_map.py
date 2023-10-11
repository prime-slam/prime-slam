from itertools import compress
from typing import Dict

import numpy as np

from src.frame import Frame
from src.mapping.landmark.landmark import Landmark
from src.mapping.landmark.landmark_creator.point_landmark_creator import (
    PointLandmarkCreator,
)
from src.mapping.map import Map
from src.projection import Projector


class PointMap(Map):
    def __init__(self, projector: Projector, landmarks: Dict[int, Landmark] = None):
        super().__init__(projector, PointLandmarkCreator(), landmarks)

    def get_visible_map(
        self,
        frame: Frame,
    ):
        visible_map = PointMap(self._projector)
        landmark_positions = self.get_positions()
        landmark_positions_cam = self._projector.transform(
            landmark_positions, frame.world_to_camera_transform
        )
        map_mean_viewing_directions = self.get_mean_viewing_directions()
        projected_map = self._projector.project(
            landmark_positions_cam,
            frame.sensor_measurement.depth.intrinsics,
            np.eye(4),
        )
        depth_mask = landmark_positions_cam[:, 2] > 0
        origin = frame.origin
        viewing_directions = landmark_positions - origin
        viewing_directions = viewing_directions / np.linalg.norm(
            viewing_directions, axis=-1
        ).reshape(-1, 1)
        height, width = frame.sensor_measurement.depth.depth_map.shape[:2]

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
        visible_landmarks = list(compress(self._landmarks.values(), mask))
        visible_map.add_landmarks(visible_landmarks)

        return visible_map
