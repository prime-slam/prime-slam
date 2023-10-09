import cv2

from src.observation.detection.points.opencv_point_detector import OpenCVPointDetector

__all__ = ["SIFT"]


class SIFT(OpenCVPointDetector):
    def __init__(self):
        super().__init__(cv2.SIFT_create())  # TODO
