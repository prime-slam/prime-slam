import cv2
import numpy as np

from typing import List

from prime_slam.observation.detection.detector import Detector
from prime_slam.observation.keyobject import Keyobject
from prime_slam.observation.opencv_keypoint import OpenCVKeypoint
from prime_slam.sensor.rgbd import RGBDImage

__all__ = ["OpenCVPointDetector"]


class OpenCVPointDetector(Detector):
    def __init__(self, detector, squared_sigma_levels: np.ndarray):
        self.detector = detector
        self.squared_sigma_levels = squared_sigma_levels
        self.inv_squared_sigma_levels = 1 / squared_sigma_levels

    def detect(self, sensor_data: RGBDImage) -> List[Keyobject]:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        keypoints = self.detector.detect(gray, None)

        return [
            OpenCVKeypoint(kp, self.inv_squared_sigma_levels[kp.octave])
            for kp in keypoints
        ]
