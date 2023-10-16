import numpy as np

from dataclasses import dataclass
from typing import Callable, Any

from prime_slam.slam.config.slam_config import SLAMConfig
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator
from prime_slam.projection.projector import Projector

__all__ = ["TrackingConfig"]


@dataclass
class TrackingConfig:
    matcher: Callable[[np.ndarray, np.ndarray], np.ndarray]
    projector: Projector
    pose_estimator: PoseEstimator
    observation_name: str

    @classmethod
    def from_slam_config(cls, slam_config: SLAMConfig):
        return cls(
            matcher=slam_config.matcher,
            projector=slam_config.projector,
            pose_estimator=slam_config.pose_estimator,
            observation_name=slam_config.observation_name,
        )
