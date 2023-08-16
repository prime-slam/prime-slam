import cv2
import numpy as np

from typing import List

from src.description.descriptor_base import Descriptor
from src.observation.keyobject import Keyobject
from src.sensor.rgbd import RGBDImage


class OpenCVPointDescriptor(Descriptor):
    def __init__(self, descriptor):
        self.descriptor = descriptor

    def descript(
        self, keypoints: List[Keyobject], sensor_data: RGBDImage
    ) -> np.ndarray:
        gray = cv2.cvtColor(np.array(sensor_data.rgb.image), cv2.COLOR_RGB2GRAY)
        keypoints = [keypoint.data for keypoint in keypoints]
        _, descriptors = self.descriptor.compute(gray, keypoints)

        return descriptors
