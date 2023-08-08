import cv2
import numpy as np

from typing import List

from src.observation.point_observation import PointObservation
from src.matching.matcher_base import Matcher


class SIFTMatcher(Matcher):
    def __init__(self, ratio_threshold=0.7):
        self.sift = cv2.SIFT_create()
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        first_observations: List[PointObservation],
        second_observations: List[PointObservation],
        first_sensor_data,
        second_sensor_data,
    ):
        first_descriptors = np.array(
            [observation.descriptor for observation in first_observations]
        )
        second_descriptors = np.array(
            [observation.descriptor for observation in second_observations]
        )
        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(first_descriptors, second_descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
        return np.array(good_matches)
