import numpy as np

from src.sensor.sensor_data_base import SensorData


class DepthImage(SensorData):
    def __init__(
        self,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        depth_scale: float,
    ):
        self.depth_map = depth_map
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale
        self.bf = 400
