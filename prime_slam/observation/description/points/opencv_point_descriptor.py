import cv2
import numpy as np

from typing import List

from prime_slam.observation.description.descriptor import Descriptor
from prime_slam.observation.keyobject import Keyobject
from prime_slam.sensor.sensor_data import SensorData
from prime_slam.typing.hints import ArrayNxM

__all__ = ["OpenCVPointDescriptor"]


class OpenCVPointDescriptor(Descriptor):
    def __init__(self, descriptor):
        self.descriptor = descriptor

    def descript(
        self, keypoints: List[Keyobject], sensor_data: SensorData
    ) -> ArrayNxM[float]:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        keypoints = [keypoint.data for keypoint in keypoints]
        _, descriptors = self.descriptor.compute(gray, keypoints)

        return descriptors
