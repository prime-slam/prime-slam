import numpy as np

from src.observation.mask.coordinates_mask import CoordinatesMask
from src.typing import ArrayN, ArrayNxM


class PointDepthFilter(CoordinatesMask):
    def create(self, coordinates: ArrayNxM[float], sensor_data) -> ArrayN[bool]:
        coordinates = coordinates.astype(int)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        mask = sensor_data.depth.depth_map[y, x] > 0

        return mask
