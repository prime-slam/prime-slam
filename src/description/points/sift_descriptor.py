import cv2

from src.description.points.opencv_point_descriptor import OpenCVPointDescriptor


class SIFTDescriptor(OpenCVPointDescriptor):
    def __init__(self):
        super().__init__(cv2.SIFT_create())
