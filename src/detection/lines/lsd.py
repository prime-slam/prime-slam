import cv2
import numpy as np

from typing import List

from src.feature.line_feature import LineFeature
from src.sensor.rgbd import RGBDImage


class LSD:
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

    def extract(self, sensor_data: RGBDImage) -> List[LineFeature]:
        opencv_image = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)

        lines, _, _, _ = self.lsd.detect(opencv_image)
        lines = lines.flatten().reshape(-1, 4)
        features = []
        for x1, y1, x2, y2 in zip(lines):
            features.append(LineFeature(np.array([x1, y1]), np.array([x2, y2])))
        return features
