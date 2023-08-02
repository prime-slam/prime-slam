import cv2

from src.detection.points.opencv_point_detector import OpenCVPointDetector


class ORB(OpenCVPointDetector):
    def __init__(self, nfeatures):
        super().__init__(cv2.ORB_create(nfeatures))
