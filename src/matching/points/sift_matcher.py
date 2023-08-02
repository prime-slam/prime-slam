import cv2
import numpy as np

from typing import List

from src.feature.feature import PointFeature
from src.matching.matcher_base import Matcher


class SIFTMatcher(Matcher):
    def __init__(self, ratio_threshold=0.7):
        self.sift = cv2.SIFT_create()
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        first_features: List[PointFeature],
        second_features: List[PointFeature],
        first_sensor_data,
        second_sensor_data,
    ):
        first_descriptors = np.array([feature.descriptor for feature in first_features])
        second_descriptors = np.array([feature.descriptor for feature in second_features])
        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(first_descriptors, second_descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
        return np.array(good_matches)
