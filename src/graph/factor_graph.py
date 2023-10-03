from typing import List

from src.graph.factor.observation.observation_factor import ObservationFactor
from src.graph.factor.odometry_factor import OdometryFactor
from src.graph.node.landmark_node import LandmarkNode
from src.graph.node.pose_node import PoseNode


class FactorGraph:
    def __init__(self):
        self.nodes = []
        self.pose_nodes = []
        self._landmark_nodes = {}
        self.factors = []
        self.odometry_factors = []
        self.observation_factors: List[ObservationFactor] = []

    def add_pose_node(self, node_id, pose):
        new_pose_node = PoseNode(node_id, pose)
        self.nodes.append(new_pose_node)
        self.pose_nodes.append(new_pose_node)

    def set_not_bad(self, landmark_id):
        self._landmark_nodes[landmark_id].set_not_bad()

    @property
    def landmark_nodes(self):
        return list(self._landmark_nodes.values())

    def add_landmark_node(self, node_id, position):
        new_landmark_node = LandmarkNode(node_id, position)
        self.nodes.append(new_landmark_node)
        self._landmark_nodes[node_id] = new_landmark_node

    def add_odometry_factor(self, from_id: int, to_id: int, relative_pose):
        odometry_factor = OdometryFactor(relative_pose, from_id, to_id)
        self.factors.append(odometry_factor)
        self.odometry_factors.append(odometry_factor)

    def update_landmarks_positions(self, new_positions):
        for new_position, landmark_node in zip(
            new_positions, self._landmark_nodes.values()
        ):
            landmark_node.position = new_position

    def update_poses(self, new_poses):
        for new_pose, pose_node in zip(new_poses, self.pose_nodes):
            pose_node.pose = new_pose

    def add_observation_factor(
        self, pose_id, landmark_id, observation, sensor_measurement, information=None
    ):
        observation_factor = ObservationFactor(
            keyobject_position_2d=observation,
            pose_node=pose_id,
            landmark_node=landmark_id,
            sensor_measurement=sensor_measurement,
            information=information,
        )
        self.factors.append(observation_factor)
        self.observation_factors.append(observation_factor)