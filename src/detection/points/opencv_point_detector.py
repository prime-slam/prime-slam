import cv2
import numpy as np

from typing import List

from src.detection.detector_base import Detector
from src.feature.point_feature import PointFeature
from src.sensor.rgbd import RGBDImage


class OpenCVPointDetector(Detector):
    def __init__(self, detector):
        self.detector = detector

    def detect(self, sensor_data: RGBDImage) -> List[PointFeature]:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        keypoints = cv2.KeyPoint.convert(keypoints)
        features = []
        for (x, y), desc in zip(keypoints, descriptors):
            features.append(PointFeature(x, y, desc))
        return features
