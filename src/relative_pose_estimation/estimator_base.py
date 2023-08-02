from abc import ABC, abstractmethod

from src.keyframe import Keyframe


class PoseEstimatorBase(ABC):
    @abstractmethod
    def estimate(self, new_keyframe: Keyframe, prev_keyframe: Keyframe, matches):
        pass
