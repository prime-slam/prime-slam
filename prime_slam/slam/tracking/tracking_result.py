from dataclasses import dataclass

from prime_slam.slam.tracking.data_association import DataAssociation
from prime_slam.geometry.pose import Pose

__all__ = ["TrackingResult"]


@dataclass
class TrackingResult:
    pose: Pose
    associations: DataAssociation
