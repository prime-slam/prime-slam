import cv2

from src.description.points.opencv_point_descriptor import OpenCVPointDescriptor


class ORBDescriptor(OpenCVPointDescriptor):
    def __init__(self):
        super().__init__(cv2.ORB_create())
