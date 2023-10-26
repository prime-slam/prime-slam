# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from prime_slam.projection.projector import Projector
from prime_slam.typing.hints import Transformation

__all__ = ["PointProjector"]


class PointProjector(Projector):
    def transform(self, points_3d, transformation_matrix: Transformation):
        ones_column = np.ones((len(points_3d), 1))
        points_3d_homo = np.concatenate([points_3d, ones_column], axis=1)
        transformed_points_3d_homo = transformation_matrix @ points_3d_homo.T
        transformed_points_3d_homo /= transformed_points_3d_homo[3]

        return transformed_points_3d_homo.T[..., :3]

    def back_project(
        self,
        points_2d,
        depth_map: np.ndarray,
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
        z_3d = depths
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
