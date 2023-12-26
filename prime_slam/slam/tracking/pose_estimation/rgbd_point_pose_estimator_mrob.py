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

import g2o
import mrob
import numpy as np

from prime_slam.geometry.pose import Pose
from prime_slam.observation.observations_batch import ObservationData
from prime_slam.projection.point_projector import PointProjector
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator
from prime_slam.typing.hints import ArrayNx2, ArrayNx3

__all__ = ["RGBDPointPoseEstimatorMROB"]


class RGBDPointPoseEstimatorMROB(PoseEstimator):
    def __init__(
        self,
        camera_intrinsics,
    ):
        self.camera_k = np.array(
            [
                camera_intrinsics[0, 0],
                camera_intrinsics[1, 1],
                camera_intrinsics[0, 2],
                camera_intrinsics[1, 2],
            ]
        )

        self.projector = PointProjector()

    def estimate_absolute_pose(
        self,
        new_observation_data: ObservationData,
        map_3d_points: ArrayNx3[float],
        matches: ArrayNx2[float],
    ) -> Pose:
        new_keypoints = new_observation_data.coordinates
        new_keypoints_index = matches[:, 0]
        map_points_index = matches[:, 1]

        kpts_obs = new_keypoints[new_keypoints_index]
        map_3d_points = map_3d_points[map_points_index]

        nan_mask = np.logical_or.reduce(
            np.isinf(map_3d_points) | np.isnan(map_3d_points),
            axis=-1,
        )

        kpts_obs = kpts_obs[~nan_mask]
        map_3d_points = map_3d_points[~nan_mask]
        mrob_graph = mrob.FGraph(mrob.HUBER)

        node_id = mrob_graph.add_node_pose_3d(mrob.geometry.SE3())

        for keypoint_coords, landmark_coords in zip(kpts_obs, map_3d_points):
            # create a landmark point
            landmark_id = mrob_graph.add_node_landmark_3d(
                landmark_coords, mrob.NODE_ANCHOR
            )
            mrob_graph.add_factor_camera_proj_3d_point(
                obs=keypoint_coords,
                nodePoseId=node_id,
                nodeLandmarkId=landmark_id,
                camera_k=self.camera_k,
                obsInvCov=np.eye(2),
            )
        mrob_graph.solve(mrob.LM, maxIters=20)
        T_estim = mrob.geometry.SE3(mrob_graph.get_estimated_state()[0]).inv()
        return Pose(T_estim.T())  # prev to new

    def estimate_relative_pose(
        self,
        new_observation_data: ObservationData,
        prev_observation_data: ObservationData,
        matches: ArrayNx2[float],
    ) -> Pose:
        prev_keypoints = prev_observation_data.coordinates
        prev_depth_map = prev_observation_data.sensor_measurement.depth.depth_map
        prev_intrinsics = prev_observation_data.sensor_measurement.depth.intrinsics
        prev_keypoints_3d = self.projector.back_project(
            prev_keypoints, prev_depth_map, prev_intrinsics, np.eye(4)
        )

        return self.estimate_absolute_pose(
            new_observation_data, prev_keypoints_3d, matches
        )
