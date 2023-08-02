import numpy as np
import pytlbd
from skimage.feature import match_descriptors
from typing import List

from src.feature.line_feature import LineFeature
from src.matching.matcher_base import Matcher
from src.sensor.rgbd import RGBDImage


class LBD(Matcher):
    def __init__(self, bands_number=9, band_width=7):
        self.bands_number = bands_number
        self.band_width = band_width

    def match(
        self,
        first_features: List[LineFeature],
        second_features: List[LineFeature],
        first_sensor_data: RGBDImage,
        second_sensor_data: RGBDImage,
    ):
        first_lines = (
            np.array(
                [[feature.start_point, feature.end_point] for feature in first_features]
            )
            .flatten()
            .reshape(-1, 4)
        )
        second_lines = (
            np.array(
                [
                    [feature.start_point, feature.end_point]
                    for feature in second_features
                ]
            )
            .flatten()
            .reshape(-1, 4)
        )

        first_descriptors = pytlbd.lbd_single_scale(
            first_sensor_data.rgb, first_lines, self.bands_number, self.band_width
        )
        second_descriptors = pytlbd.lbd_single_scale(
            first_sensor_data.rgb, second_lines, self.bands_number, self.band_width
        )

        return match_descriptors(first_descriptors, second_descriptors)
