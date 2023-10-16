import numpy as np

from prime_slam.sensor.sensor_data import SensorData

__all__ = ["RGBImage"]


class RGBImage(SensorData):
    def __init__(
        self,
        image: np.ndarray,
    ):
        self.image = image
