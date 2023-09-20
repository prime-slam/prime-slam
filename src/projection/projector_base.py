from abc import ABC, abstractmethod

import numpy as np


class Projector(ABC):
    @abstractmethod
    def transform(self, objects_3d, transformation_matrix):
        pass

    @abstractmethod
    def project(
        self,
        object_3d,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        pass

    @abstractmethod
    def back_project(
        self,
        object_2d,
        depth_map: np.ndarray,
        depth_scale: float,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        pass
