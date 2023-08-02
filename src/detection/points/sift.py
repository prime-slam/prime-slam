import cv2

from src.detection.points.opencv_point_detector import OpenCVPointDetector


class SIFT(OpenCVPointDetector):
    def __init__(self):
        super().__init__(cv2.SIFT_create())
