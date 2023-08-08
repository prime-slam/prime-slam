import cv2
import numpy as np
import pytlbd
from skimage.feature import match_descriptors
from typing import List

from src.observation.line_observation import LineObservation
from src.matching.matcher_base import Matcher
from src.sensor.rgbd import RGBDImage


class LBD(Matcher):
    def __init__(self, bands_number=9, band_width=7):
        self.bands_number = bands_number
        self.band_width = band_width

    def match(
        self,
        first_observations: List[LineObservation],
        second_observations: List[LineObservation],
        first_sensor_data: RGBDImage,
        second_sensor_data: RGBDImage,
    ):
        first_lines = (
            np.array(
                [[observation.start_point, observation.end_point] for observation in first_observations]
            )
            .flatten()
            .reshape(-1, 4)
        )
        second_lines = (
            np.array(
                [
                    [observation.start_point, observation.end_point]
                    for observation in second_observations
                ]
            )
            .flatten()
            .reshape(-1, 4)
        )
        first_gray_image = cv2.cvtColor(first_sensor_data.rgb.image, cv2.COLOR_RGB2GRAY)
        second_gray_image = cv2.cvtColor(
            second_sensor_data.rgb.image, cv2.COLOR_RGB2GRAY
        )
        first_descriptors = pytlbd.lbd_single_scale(
            first_gray_image, first_lines, self.bands_number, self.band_width
        )
        second_descriptors = pytlbd.lbd_single_scale(
            second_gray_image, second_lines, self.bands_number, self.band_width
        )

        return match_descriptors(first_descriptors, second_descriptors)
