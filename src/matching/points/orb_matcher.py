import cv2
import numpy as np

from typing import List

from src.observation.point_observation import PointObservation
from src.matching.matcher_base import Matcher
from src.sensor.sensor_data_base import SensorData


class ORBMatcher(Matcher):
    def __init__(self, ratio_threshold=0.7):
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        first_observations: List[PointObservation],
        second_observations: List[PointObservation],
        first_sensor_data: SensorData,
        second_sensor_data: SensorData,
    ):
        first_descs = np.array(
            [observation.descriptor for observation in first_observations]
        )
        second_descs = np.array(
            [observation.descriptor for observation in second_observations]
        )
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(first_descs, second_descs, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
        return np.array(good_matches)
