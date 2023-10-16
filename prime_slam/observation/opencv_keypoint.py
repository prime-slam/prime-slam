import math

import cv2
import numpy as np

from prime_slam.observation.keyobject import Keyobject

__all__ = ["OpenCVKeypoint"]


class OpenCVKeypoint(Keyobject):
    def __init__(self, keypoint: cv2.KeyPoint, uncertainty):
        self.keypoint = keypoint
        self._uncertainty = uncertainty

    @property
    def data(self):
        return self.keypoint

    @property
    def coordinates(self):
        return np.array(self.keypoint.pt)

    @property
    def uncertainty(self):
        return self._uncertainty
