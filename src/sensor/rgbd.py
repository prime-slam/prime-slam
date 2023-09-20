from src.sensor.depth import DepthImage
from src.sensor.rgb import RGBImage
from src.sensor.sensor_data_base import SensorData


class RGBDImage(SensorData):
    def __init__(self, rgb: RGBImage, depth: DepthImage, bf: float = 400):
        self.rgb = rgb
        self.depth = depth
        self.bf = bf
