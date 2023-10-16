import numpy as np

from collections.abc import Callable
from dataclasses import dataclass

from prime_slam.observation.description.descriptor import Descriptor
from prime_slam.observation.detection.detector import Detector
from prime_slam.observation.filter.observation_filter import ObservationsFilter
from prime_slam.projection.projector import Projector
from prime_slam.slam.mapping.map_creator.map_creator import MapCreator
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator

__all__ = ["SLAMConfig"]


@dataclass
class SLAMConfig:
    detector: Detector
    descriptor: Descriptor
    matcher: Callable[[np.ndarray, np.ndarray], np.ndarray]
    projector: Projector
    pose_estimator: PoseEstimator
    observations_filter: ObservationsFilter
    map_creator: MapCreator
    observation_name: str
