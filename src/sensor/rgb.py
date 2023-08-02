import numpy as np

from src.sensor.sensor_data_base import SensorData


class RGBImage(SensorData):
    def __init__(
        self,
        image: np.ndarray,
    ):
        self.image = image
