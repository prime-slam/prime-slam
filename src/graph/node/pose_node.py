from src.graph.node.node import Node

__all__ = ["PoseNode"]


class PoseNode(Node):
    def __init__(self, identifier, pose=None):
        super().__init__(identifier)

        self._pose = pose

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, new_pose):
        self._pose = new_pose
