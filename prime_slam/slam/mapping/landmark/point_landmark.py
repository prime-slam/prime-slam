import numpy as np

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.landmark.landmark import Landmark


class PointLandmark(Landmark):
    def __init__(self, identifier, position, feature_descriptor, keyframe: Frame):
        super().__init__(identifier, position, feature_descriptor, keyframe)

    def _calculate_viewing_directions(self, origins: np.ndarray):
        return self._position - origins
