import cv2
import numpy as np

from typing import List

from src.observation.detection.detector import Detector
from src.observation.keyobject import Keyobject
from src.observation.opencv_keypoint import OpenCVKeypoint
from src.sensor.rgbd import RGBDImage


class OpenCVPointDetector(Detector):
    def __init__(self, detector, squared_sigma_levels: np.ndarray):
        self.detector = detector
        self.squared_sigma_levels = squared_sigma_levels
        self.inv_squared_sigma_levels = 1 / squared_sigma_levels

    def detect(self, sensor_data: RGBDImage) -> List[Keyobject]:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        depth = sensor_data.depth.depth_map
        keypoints = self.detector.detect(gray, None)
        positions = np.array([kp.pt for kp in keypoints]).astype(int)

        return [
            OpenCVKeypoint(kp, self.inv_squared_sigma_levels[kp.octave])
            for kp in keypoints
        ]
