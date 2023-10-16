from prime_slam.sensor.depth import DepthImage
from prime_slam.sensor.rgb import RGBImage
from prime_slam.sensor.sensor_data import SensorData

__all__ = ["RGBDImage"]


class RGBDImage(SensorData):
    def __init__(self, rgb: RGBImage, depth: DepthImage, bf: float = 400):
        self.rgb = rgb
        self.depth = depth
        self.bf = bf
