from dataclasses import dataclass

from src.tracking.data_association import DataAssociation
from src.geometry.pose import Pose

__all__ = ["TrackingResult"]


@dataclass
class TrackingResult:
    pose: Pose
    associations: DataAssociation
