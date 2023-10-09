import cv2
import numpy as np

from src.observation.detection.points.opencv_point_detector import OpenCVPointDetector

__all__ = ["ORB"]


class ORB(OpenCVPointDetector):
    def __init__(
        self, features_number: int, scale_factor: float = 1.2, levels_number=8
    ):
        super().__init__(
            cv2.ORB_create(features_number, scale_factor, levels_number),
            self.create_squared_sigma_levels(levels_number, scale_factor),
        )

    @staticmethod
    def create_squared_sigma_levels(
        levels_number: int = 8,
        scale_factor: float = 1.2,
        init_sigma_level: float = 1.0,
        init_scale_factor: float = 1.0,
    ) -> np.ndarray:
        scale_factors = np.zeros(levels_number)
        squared_sigma_levels = np.zeros(levels_number)

        scale_factors[0] = init_scale_factor
        squared_sigma_levels[0] = init_sigma_level * init_sigma_level

        for i in range(1, levels_number):
            scale_factors[i] = scale_factors[i - 1] * scale_factor
            squared_sigma_levels[i] = (
                scale_factors[i] * scale_factors[i] * squared_sigma_levels[0]
            )

        return squared_sigma_levels
