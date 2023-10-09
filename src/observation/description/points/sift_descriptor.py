import cv2

from src.observation.description.points.opencv_point_descriptor import (
    OpenCVPointDescriptor,
)

__all__ = ["SIFTDescriptor"]


class SIFTDescriptor(OpenCVPointDescriptor):
    def __init__(self):
        super().__init__(cv2.SIFT_create())
