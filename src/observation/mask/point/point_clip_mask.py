from src.observation.mask.coordinates_mask import CoordinatesMask
from src.typing import ArrayN, ArrayNxM


class PointClipMask(CoordinatesMask):
    def create(self, coordinates: ArrayNxM[float], sensor_data) -> ArrayN[bool]:
        height, width = sensor_data.depth.depth_map.shape[:2]
        coordinates = coordinates.astype(int)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        return mask
