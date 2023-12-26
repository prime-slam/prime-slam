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
from prime_slam.projection.line_projector import LineProjector
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator
from prime_slam.typing.hints import ArrayNx2, ArrayNx3, ArrayNx6

__all__ = ["RGBDLinePoseEstimatorMROB"]


class RGBDLinePoseEstimatorMROB(PoseEstimator):
    def __init__(
        self,
        camera_intrinsics,
        reprojection_threshold=10,
        iterations_number=50,
        optimizer_iterations_number=30,
        edges_min_number=12,
    ):
        self.camera_k = np.array(
            [
                camera_intrinsics[0, 0],
                camera_intrinsics[1, 1],
                camera_intrinsics[0, 2],
                camera_intrinsics[1, 2],
            ]
        )
        self.iterations_number = iterations_number
        self.optimizer_iterations_number = optimizer_iterations_number
        self.reprojection_threshold = reprojection_threshold
        self.edges_min_number = edges_min_number
        self.projector = LineProjector()

    def create_graph(self, observations, map_3d_lines):
        mrob_graph = mrob.FGraph(mrob.HUBER)
        node_id = mrob_graph.add_node_pose_3d(mrob.geometry.SE3())

        for keyline_coords, landmark_coords in zip(observations, map_3d_lines):
            # create a landmark point
            first_endpoint_id = mrob_graph.add_node_landmark_3d(
                landmark_coords[:3], mrob.NODE_ANCHOR
            )
            second_endpoint_id = mrob_graph.add_node_landmark_3d(
                landmark_coords[3:], mrob.NODE_ANCHOR
            )
            mrob_graph.add_factor_camera_proj_3d_line(
                obsPoint1=keyline_coords[:2],
                obsPoint2=keyline_coords[2:],
                nodePoseId=node_id,
                nodePoint1=first_endpoint_id,
                nodePoint2=second_endpoint_id,
                camera_k=self.camera_k,
                obsInvCov=np.eye(2),
            )
        return mrob_graph

    def estimate_absolute_pose(
        self,
        new_observation_data: ObservationData,
        map_3d_lines: ArrayNx6[float],
        matches: ArrayNx2[float],
    ) -> Pose:
        new_keylines = new_observation_data.coordinates
        new_keylines_index = matches[:, 0]
        map_lines_index = matches[:, 1]

        observations = new_keylines[new_keylines_index]
        map_3d_lines = map_3d_lines[map_lines_index]

        nan_mask = np.logical_or.reduce(
            np.isinf(map_3d_lines) | np.isnan(map_3d_lines),
            axis=-1,
        )

        observations = observations[~nan_mask]
        map_3d_lines = map_3d_lines[~nan_mask]
        inlier_mask = np.ones(len(map_3d_lines), dtype=bool)
        mrob_graph = None
        for i in range(self.iterations_number):
            observations = observations[inlier_mask]
            map_3d_lines = map_3d_lines[inlier_mask]
            mrob_graph = self.create_graph(observations, map_3d_lines)
            mrob_graph.solve(mrob.LM, maxIters=self.optimizer_iterations_number)
            chis = mrob_graph.get_chi2_array()
            inlier_mask = chis < self.reprojection_threshold / (i + 1) ** 2
            if np.count_nonzero(inlier_mask) < self.edges_min_number:
                break

        state = mrob_graph.get_estimated_state()
        T_estim = mrob.geometry.SE3(state[0]).inv()
        return Pose(T_estim.T())  # prev to new

    def estimate_relative_pose(
        self,
        new_observation_data: ObservationData,
        prev_observation_data: ObservationData,
        matches: ArrayNx2[float],
    ) -> Pose:
        prev_keylines = prev_observation_data.coordinates
        prev_depth_map = prev_observation_data.sensor_measurement.depth.depth_map
        prev_intrinsics = prev_observation_data.sensor_measurement.depth.intrinsics
        prev_keylines_3d = self.projector.back_project(
            prev_keylines, prev_depth_map, prev_intrinsics, np.eye(4)
        )

        return self.estimate_absolute_pose(
            new_observation_data, prev_keylines_3d, matches
        )
