import cv2
import numpy as np

from typing import List

from src.detection.detector_base import Detector
from src.observation.line_observation import LineObservation
from src.sensor.rgbd import RGBDImage


class LSD(Detector):
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

    def detect(self, sensor_data: RGBDImage) -> List[LineObservation]:
        opencv_image = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)

        lines, _, _, _ = self.lsd.detect(opencv_image)
        lines = lines.flatten().reshape(-1, 4)
        observations = [
            LineObservation(np.array([x1, y1]), np.array([x2, y2]))
            for x1, y1, x2, y2 in lines
        ]

        return observations
