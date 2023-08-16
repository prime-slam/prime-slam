import cv2
import numpy as np

from typing import List

from src.detection.detector_base import Detector
from src.observation.keyobject import Keyobject
from src.observation.opencv_keypoint import OpenCVKeypoint
from src.sensor.rgbd import RGBDImage


class OpenCVPointDetector(Detector):
    def __init__(self, detector):
        self.detector = detector

    def detect(self, sensor_data: RGBDImage) -> List[Keyobject]:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        keypoints = self.detector.detect(gray, None)

        return [OpenCVKeypoint(kp) for kp in keypoints]
