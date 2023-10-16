import numpy as np

from prime_slam.geometry.util import clip_lines
from prime_slam.projection.point_projection import PointProjector
from prime_slam.projection.projector import Projector
from prime_slam.typing.hints import Transformation

__all__ = ["LineProjector"]


class LineProjector(Projector):
    def __init__(self):
        self.point_projector = PointProjector()

    def transform(self, lines_3d: np.ndarray, transformation_matrix: Transformation):
        lines_3d = lines_3d.reshape(-1, 2, 3)
        start_points_3d = lines_3d[:, 0]
        end_points_3d = lines_3d[:, 1]

        transformed_start_points_3d = self.point_projector.transform(
            start_points_3d, transformation_matrix
        )
        transformed_end_points_2d = self.point_projector.transform(
            end_points_3d, transformation_matrix
        )

        return np.column_stack([transformed_start_points_3d, transformed_end_points_2d])

    def back_project(
        self,
        lines_2d,
        depth_map: np.ndarray,
        depth_scale: float,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        height, width = depth_map.shape[:2]
        lines_2d = (
            clip_lines(lines_2d, width=width, height=height)
            .astype(int)
            .reshape(-1, 2, 2)
        )
        start_points_2d = lines_2d[:, 0]
        end_points_2d = lines_2d[:, 1]
        start_points_3d = self.point_projector.back_project(
            start_points_2d, depth_map, depth_scale, intrinsics, extrinsics
        )
        end_points_3d = self.point_projector.back_project(
            end_points_2d, depth_map, depth_scale, intrinsics, extrinsics
        )

        return np.column_stack([start_points_3d, end_points_3d])

    def project(
        self,
        lines_3d,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        start_points_3d = lines_3d[:, 0]
        end_points_3d = lines_3d[:, 1]

        projected_start_points_2d = self.point_projector.project(
            start_points_3d, intrinsics, extrinsics
        )
        projected_end_points_2d = self.point_projector.project(
            end_points_3d, intrinsics, extrinsics
        )

        return np.column_stack([projected_start_points_2d, projected_end_points_2d])
