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
import numpy as np

from typing import List

from prime_slam.slam.graph.factor.observation_factor import (
    ObservationFactor,
)
from prime_slam.slam.graph.factor_graph import FactorGraph
from prime_slam.slam.graph.node.landmark_node import LandmarkNode
from prime_slam.slam.graph.node.pose_node import PoseNode
from prime_slam.slam.backend.backend import Backend

__all__ = ["G2OPointSLAMBackend"]


class G2OPointSLAMBackend(Backend):
    def __init__(self, intrinsics, optimizer_iterations_number=25):
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        self.optimizer_iterations_number = optimizer_iterations_number
        self.bf = 400

    def optimize(self, graph: FactorGraph, verbose=False):
        optimizer = self.__create_optimizer()
        pose_nodes: List[PoseNode] = graph.pose_nodes
        landmark_nodes: List[LandmarkNode] = graph.landmark_nodes
        observation_factors: List[ObservationFactor] = graph.observation_factors
        pose_vertices = []
        landmark_vertices = []
        edges = []
        max_id = 0
        for pose_node in pose_nodes:
            vertex = g2o.VertexSE3Expmap()
            pose = pose_node.pose
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            se3 = g2o.SE3Quat(rotation, translation)
            vertex.set_estimate(se3)
            vertex.set_id(pose_node.identifier)
            vertex.set_fixed(pose_node.identifier == 0)
            optimizer.add_vertex(vertex)
            pose_vertices.append(vertex)
            max_id = max(max_id, pose_node.identifier)
        max_id = max_id + 1
        for landmark_node in landmark_nodes:
            vertex = g2o.VertexSBAPointXYZ()
            vertex_id = max_id + landmark_node.identifier
            vertex.set_estimate(landmark_node.position)
            vertex.set_id(vertex_id)
            vertex.set_marginalized(True)
            vertex.set_fixed(False)
            optimizer.add_vertex(vertex)
            landmark_vertices.append(vertex)

        for observation_factor in observation_factors:
            landmark_id = observation_factor.to_node + max_id
            pose_id = observation_factor.from_node
            edge = g2o.EdgeStereoSE3ProjectXYZ()

            edge.set_vertex(0, optimizer.vertex(landmark_id))
            edge.set_vertex(1, optimizer.vertex(pose_id))
            stereo_observation = self.__convert_to_stereo(
                observation_factor.observation,
                observation_factor.depth_map / observation_factor.depth_scale,
                self.bf,
            )
            edge.set_measurement(stereo_observation)
            edge.set_information(np.eye(3) * observation_factor.information)
            kernel = g2o.RobustKernelHuber()
            edge.set_robust_kernel(kernel)

            edge.fx = self.fx
            edge.fy = self.fy
            edge.cx = self.cx
            edge.cy = self.cy
            edge.bf = self.bf
            optimizer.add_edge(edge)
            edges.append(edge)

        optimizer.set_verbose(verbose)
        optimizer.initialize_optimization()
        optimizer.optimize(self.optimizer_iterations_number)

        new_poses = np.array([v.estimate().matrix() for v in pose_vertices])
        new_points = np.array([v.estimate() for v in landmark_vertices])

        return new_poses, new_points

    @staticmethod
    def __convert_to_stereo(coords, depth_map: np.ndarray, bf):
        x, y = coords
        d = depth_map[int(y), int(x)]
        return np.array([x, y, x - bf / d])

    @staticmethod
    def __create_optimizer():
        optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)

        return optimizer
