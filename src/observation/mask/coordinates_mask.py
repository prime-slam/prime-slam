from abc import ABC, abstractmethod

from src.typing import ArrayN, ArrayNxM


class CoordinatesMask(ABC):
    @abstractmethod
    def create(self, coordinates: ArrayNxM[float], sensor_data) -> ArrayN[bool]:
        pass
