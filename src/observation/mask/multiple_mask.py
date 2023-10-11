import numpy as np

from typing import List

from src.observation.mask.coordinates_mask import CoordinatesMask
from src.typing import ArrayN, ArrayNxM


class CoordinatesMultipleMask(CoordinatesMask):
    def __init__(self, observation_masks: List[CoordinatesMask]):
        self.observation_masks = observation_masks

    def create(self, coordinates: ArrayNxM[float], sensor_data) -> ArrayN[bool]:
        mask = self.observation_masks[0].create(coordinates, sensor_data)
        for i in range(1, len(self.observation_masks)):
            nonzero_indices = mask.nonzero()[0]
            new_mask = self.observation_masks[i].create(coordinates[mask], sensor_data)
            new_zero_indices = nonzero_indices[~new_mask]
            mask[new_zero_indices] = False

        return mask
