import cv2
import numpy as np

from src.detection.points.opencv_point_detector import OpenCVPointDetector


class ORB(OpenCVPointDetector):
    def __init__(self, nfeatures, scale_factor=1.2, num_levels=8):
        super().__init__(
            cv2.ORB_create(nfeatures, scale_factor, num_levels),
            self.get_squared_sigma_levels(num_levels, scale_factor),
        )

    @staticmethod
    def get_squared_sigma_levels(
        num_levels=8, scale_factor=1.2, init_sigma_level=1.0, init_scale_factor=1.0
    ):
        scale_factors = np.zeros(num_levels)
        squared_sigma_levels = np.zeros(num_levels)

        scale_factors[0] = init_scale_factor
        squared_sigma_levels[0] = init_sigma_level * init_sigma_level

        for i in range(1, num_levels):
            scale_factors[i] = scale_factors[i - 1] * scale_factor
            squared_sigma_levels[i] = (
                scale_factors[i] * scale_factors[i] * squared_sigma_levels[0]
            )

        return squared_sigma_levels
