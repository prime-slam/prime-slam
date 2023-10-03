from abc import ABC, abstractmethod

from src.frame import Frame


class PoseEstimator(ABC):
    @abstractmethod
    def estimate_relative_pose(
        self, new_keyframe: Frame, prev_keyframe: Frame, matches, idx
    ):
        pass

    @abstractmethod
    def estimate_absolute_pose(self, new_keyframe: Frame, map_3d_objects, matches, idx):
        pass
