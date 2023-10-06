import numpy as np

from dataclasses import dataclass

from src.data_association import DataAssociation


@dataclass
class TrackingResult:
    pose: np.ndarray
    associations: DataAssociation
