import numpy as np

from typing import Optional

from src.geometry.transform import make_euclidean_transform
from src.typing.hints import Rotation, Translation, Transformation


class Pose:
    def __init__(self, transformation: Optional[Transformation] = None):
        self._transform: Transformation = None
        self._transform_inv: Transformation = None
        self._rotation = None
        self._translation = None
        self._rotation_inv = None
        self._translation_inv = None
        if transformation is not None:
            self.update(transformation)

    def update(self, new_transformation: Transformation) -> None:
        self._transform: Transformation = new_transformation
        self._transform_inv: Transformation = np.linalg.inv(self._transform)
        self._rotation = self._transform[:3, :3]
        self._translation = self._transform[:3, 3]

    def inverse(self) -> "Pose":
        return Pose(np.linalg.inv(self._transform))

    @property
    def transformation(self):
        return self._transform

    @property
    def rotation(self):
        return self._rotation

    @property
    def translation(self):
        return self._translation

    @classmethod
    def from_rotation_and_translation(
        cls, rotation: Rotation, translation: Translation
    ):
        return cls(make_euclidean_transform(rotation, translation))
