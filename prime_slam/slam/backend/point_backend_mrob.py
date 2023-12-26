import mrob
import numpy as np

from typing import List

from prime_slam.slam.backend.backend import Backend
from prime_slam.slam.graph.factor.observation_factor import ObservationFactor
from prime_slam.slam.graph.factor_graph import FactorGraph
from prime_slam.slam.graph.node.landmark_node import LandmarkNode
from prime_slam.slam.graph.node.pose_node import PoseNode

__all__ = ["MROBPointBackend"]


class MROBPointBackend(Backend):
    def __init__(self, intrinsics, optimizer_iterations_number=50):
        self.camera_k = np.array(
            [
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            ]
        )
        self.optimizer_iterations_number = optimizer_iterations_number

    def optimize(self, graph: FactorGraph, verbose=False):
        mrob_graph = mrob.FGraph(mrob.HUBER)
        pose_nodes: List[PoseNode] = graph.pose_nodes
        landmark_nodes: List[LandmarkNode] = graph.landmark_nodes
        observation_factors: List[ObservationFactor] = graph.observation_factors
        pose_id_to_mrob_id = {}
        landmark_id_to_mrob_id = {}

        for pose_node in pose_nodes:
            pose_id = mrob_graph.add_node_pose_3d(
                mrob.SE3(pose_node.pose),
                mrob.NODE_ANCHOR if pose_node.identifier == 0 else mrob.NODE_STANDARD,
            )
            pose_id_to_mrob_id[pose_node.identifier] = pose_id

        for landmark_node in landmark_nodes:
            landmark_id = mrob_graph.add_node_landmark_3d(
                landmark_node.position, mrob.NODE_STANDARD
            )
            landmark_id_to_mrob_id[landmark_node.identifier] = landmark_id

        for observation_factor in observation_factors:
            landmark_id = landmark_id_to_mrob_id[observation_factor.to_node]
            pose_id = pose_id_to_mrob_id[observation_factor.from_node]
            mrob_graph.add_factor_camera_proj_3d_point(
                obs=observation_factor.observation,
                nodePoseId=pose_id,
                nodeLandmarkId=landmark_id,
                camera_k=self.camera_k,
                obsInvCov=np.eye(2),
            )
        mrob_graph.solve(
            mrob.LM, maxIters=self.optimizer_iterations_number, verbose=verbose
        )
        estimated_state = mrob_graph.get_estimated_state()
        new_points = np.array(
            [
                estimated_state[landmark_id].reshape(3)
                for landmark_id in landmark_id_to_mrob_id.values()
            ]
        )
        new_poses = np.array(
            [
                mrob.geometry.SE3(estimated_state[pose_id]).T()
                for pose_id in pose_id_to_mrob_id.values()
            ]
        )

        return new_poses, new_points
