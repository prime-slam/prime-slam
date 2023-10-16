import numpy as np

from prime_slam.sensor.sensor_data import SensorData

__all__ = ["DepthImage"]


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
