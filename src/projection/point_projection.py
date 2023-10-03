import numpy as np

from src.projection.projector_base import Projector


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