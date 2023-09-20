from src.graph.factor.factor_base import Factor


class OdometryFactor(Factor):
    def __init__(self, relative_pose, from_node, to_node, information=None):
        super().__init__(from_node, to_node, information)
        self._relative_pose = relative_pose

    @property
    def relative_pose(self):
        return self._relative_pose

    @relative_pose.setter
    def relative_pose(self, new_relative_pose):
        self._relative_pose = new_relative_pose
