from src.keyframe import Keyframe
from src.relative_pose_estimation.estimator_base import PoseEstimatorBase


class RelativePoseEstimator(PoseEstimatorBase):
    def estimate(self, new_keyframe: Keyframe, prev_keyframe: Keyframe, matches):
        pass
