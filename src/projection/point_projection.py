from itertools import compress

import numpy as np

from src.frame import Frame
from src.mapping.map import Map
from src.projection.projector import Projector

__all__ = ["PointProjector"]


class PointProjector(Projector):
    def transform(self, points_3d, transformation_matrix):
        ones_column = np.ones((len(points_3d), 1))
        points_3d_homo = np.concatenate([points_3d, ones_column], axis=1)
        transformed_points_3d_homo = transformation_matrix @ points_3d_homo.T
        transformed_points_3d_homo /= transformed_points_3d_homo[3]

        return transformed_points_3d_homo.T[..., :3]

    def back_project(
        self,
        points_2d,
        depth_map: np.ndarray,
        depth_scale: float,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        x, y = points_2d[:, 0].astype(int), points_2d[:, 1].astype(int)
        depths = depth_map[y, x]
        nonzero_depths = depths != 0
        z_3d = depths / depth_scale
        x_3d = np.zeros(len(z_3d))
        y_3d = np.zeros(len(z_3d))
        x_3d[nonzero_depths] = (x[nonzero_depths] - cx) / fx * z_3d[nonzero_depths]
        y_3d[nonzero_depths] = (y[nonzero_depths] - cy) / fy * z_3d[nonzero_depths]
        # set np.nan if depth is zero
        z_3d[~nonzero_depths] = np.nan
        x_3d[~nonzero_depths] = np.nan
        y_3d[~nonzero_depths] = np.nan

        return self.transform(
            np.column_stack([x_3d, y_3d, z_3d]), np.linalg.inv(extrinsics)
        )

    def project(
        self,
        points_3d,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        projection_matrix = intrinsics @ extrinsics
        ones_column = np.ones((len(points_3d), 1))
        points_homo = np.concatenate([points_3d, ones_column], axis=1)
        projected_points_homo = projection_matrix @ points_homo.T
        projected_points_homo /= projected_points_homo[2]
        projected_points = projected_points_homo.T[..., :2]

        return projected_points

    def get_visible_map(
        self,
        landmarks_map: Map,
        frame: Frame,
    ):
        visible_map = Map()
        landmark_positions = landmarks_map.get_positions()
        landmark_positions_cam = self.transform(
            landmark_positions, frame.world_to_camera_transform
        )
        map_mean_viewing_directions = landmarks_map.get_mean_viewing_directions()
        projected_map = self.project(
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
        visible_landmarks = list(compress(landmarks_map.get_landmarks(), mask))
        visible_map.add_landmarks(visible_landmarks)

        return visible_map
