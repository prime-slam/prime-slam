from abc import ABC, abstractmethod

from src.keyframe import Keyframe


class PoseEstimator(ABC):
    @abstractmethod
    def estimate_relative_pose(
        self, new_keyframe: Keyframe, prev_keyframe: Keyframe, matches, idx
    ):
        pass

    @abstractmethod
    def estimate_absolute_pose(
        self, new_keyframe: Keyframe, map_3d_objects, matches, idx
    ):
        pass
