import mrob
import numpy as np

from itertools import compress
from typing import List

from prime_slam.slam.backend.backend import Backend
from prime_slam.slam.graph.factor.observation_factor import ObservationFactor
from prime_slam.slam.graph.factor_graph import FactorGraph
from prime_slam.slam.graph.node.landmark_node import LandmarkNode
from prime_slam.slam.graph.node.pose_node import PoseNode

__all__ = ["MROBLineBackend"]


class MROBLineBackend(Backend):
    def __init__(
        self,
        intrinsics,
        reprojection_threshold=10,
        iterations_number=50,
        optimizer_iterations_number=5,
        edges_min_number=12,
    ):
        self.camera_k = np.array(
            [
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            ]
        )
        self.iterations_number = iterations_number
        self.optimizer_iterations_number = optimizer_iterations_number
        self.reprojection_threshold = reprojection_threshold
        self.edges_min_number = edges_min_number

    def create_graph(
        self,
        pose_nodes: List[PoseNode],
        landmark_nodes: List[LandmarkNode],
        observation_factors: List[ObservationFactor],
    ):
        mrob_graph = mrob.FGraph(mrob.HUBER)
        pose_id_to_mrob_id = {}
        landmark_id_to_mrob_id = {}

        for pose_node in pose_nodes:
            pose_id = mrob_graph.add_node_pose_3d(
                mrob.SE3(pose_node.pose),
                mrob.NODE_ANCHOR if pose_node.identifier == 0 else mrob.NODE_STANDARD,
            )
            pose_id_to_mrob_id[pose_node.identifier] = pose_id

        for landmark_node in landmark_nodes:
            landmark_coords = landmark_node.position
            first_endpoint_id = mrob_graph.add_node_landmark_3d(
                landmark_coords[:3], mrob.NODE_STANDARD
            )
            second_endpoint_id = mrob_graph.add_node_landmark_3d(
                landmark_coords[3:], mrob.NODE_STANDARD
            )
            landmark_id_to_mrob_id[landmark_node.identifier] = (
                first_endpoint_id,
                second_endpoint_id,
            )

        for observation_factor in observation_factors:
            keyline_coords = observation_factor.observation
            first_endpoint_id, second_endpoint_id = landmark_id_to_mrob_id[
                observation_factor.to_node
            ]
            pose_id = pose_id_to_mrob_id[observation_factor.from_node]
            mrob_graph.add_factor_camera_proj_3d_line(
                obsPoint1=keyline_coords[:2],
                obsPoint2=keyline_coords[2:],
                nodePoseId=pose_id,
                nodePoint1=first_endpoint_id,
                nodePoint2=second_endpoint_id,
                camera_k=self.camera_k,
                obsInvCov=np.eye(2),
            )
        return mrob_graph, pose_id_to_mrob_id, landmark_id_to_mrob_id

    def optimize(self, graph: FactorGraph, verbose=True):
        pose_nodes: List[PoseNode] = graph.pose_nodes
        landmark_nodes: List[LandmarkNode] = graph.landmark_nodes
        observation_factors: List[ObservationFactor] = graph.observation_factors
        mrob_graph, pose_id_to_mrob_id, landmark_id_to_mrob_id = self.create_graph(
            pose_nodes, landmark_nodes, observation_factors
        )
        inlier_mask = np.ones(len(observation_factors), dtype=bool)
        mrob_graph = None
        for i in range(self.iterations_number):
            observation_factors = list(compress(observation_factors, inlier_mask))
            mrob_graph, pose_id_to_mrob_id, landmark_id_to_mrob_id = self.create_graph(
                pose_nodes, landmark_nodes, observation_factors
            )
            mrob_graph.solve(mrob.LM, maxIters=self.optimizer_iterations_number)
            chis = mrob_graph.get_chi2_array()
            inlier_mask = chis < self.reprojection_threshold / (i + 1) ** 2

        estimated_state = mrob_graph.get_estimated_state()
        new_lines = np.array(
            [
                np.concatenate(
                    [
                        estimated_state[first_endpoint_id].reshape(3),
                        estimated_state[second_endpoint_id].reshape(3),
                    ]
                )
                for first_endpoint_id, second_endpoint_id in landmark_id_to_mrob_id.values()
            ]
        )
        new_poses = np.array(
            [
                mrob.geometry.SE3(estimated_state[pose_id]).T()
                for pose_id in pose_id_to_mrob_id.values()
            ]
        )

        return new_poses, new_lines
