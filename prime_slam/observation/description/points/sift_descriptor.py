import cv2

from prime_slam.observation.description.points.opencv_point_descriptor import (
    OpenCVPointDescriptor,
)

__all__ = ["SIFTDescriptor"]


class SIFTDescriptor(OpenCVPointDescriptor):
    def __init__(self, octave_layers_number: int = 3):
        super().__init__(cv2.SIFT_create(nOctaveLayers=octave_layers_number))
