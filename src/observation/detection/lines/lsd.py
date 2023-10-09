import cv2
import numpy as np

from typing import List

from src.observation.detection.detector import Detector
from src.observation.keyline import Keyline
from src.observation.keyobject import Keyobject
from src.sensor.rgbd import RGBDImage

__all__ = ["LSD"]


class LSD(Detector):
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

    def detect(self, sensor_data: RGBDImage) -> List[Keyobject]:
        opencv_image = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)

        lines, _, _, scores = self.lsd.detect(opencv_image)
        lines = lines.flatten().reshape(-1, 4)
        observations = [
            Keyline(x1, y1, x2, y2, uncertainty=score)
            for (x1, y1, x2, y2), score in zip(lines, scores)
        ]

        return observations
