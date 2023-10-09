import cv2
import numpy as np
import pytlbd

from typing import List

from src.observation.description.descriptor import Descriptor
from src.observation.keyobject import Keyobject
from src.sensor.rgbd import RGBDImage

__all__ = ["LBD"]


class LBD(Descriptor):
    def __init__(self, bands_number: int = 9, band_width: int = 7):
        self.bands_number = bands_number
        self.band_width = band_width

    def descript(
        self,
        keylines: List[Keyobject],
        sensor_data: RGBDImage,
    ) -> np.ndarray:
        lines = (
            np.array([keyline.data for keyline in keylines]).flatten().reshape(-1, 4)
        )
        gray_image = cv2.cvtColor(sensor_data.rgb.image, cv2.COLOR_RGB2GRAY)
        descriptors = pytlbd.lbd_single_scale(
            gray_image, lines, self.bands_number, self.band_width
        )

        return descriptors
