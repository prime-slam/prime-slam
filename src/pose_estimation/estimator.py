from abc import ABC, abstractmethod

# from src.frame import Frame
from src.geometry.pose import Pose

__all__ = ["PoseEstimator"]


class PoseEstimator(ABC):
    @abstractmethod
    def estimate_relative_pose(self, new_keyframe, prev_keyframe, matches, idx) -> Pose:
        pass

    @abstractmethod
    def estimate_absolute_pose(
        self, new_keyframe, map_3d_objects, matches, idx
    ) -> Pose:
        pass
