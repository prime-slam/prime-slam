import cv2
import numpy as np

from typing import List

from src.feature.point_feature import PointFeature
from src.matching.matcher_base import Matcher
from src.sensor.sensor_data_base import SensorData


class ORBMatcher(Matcher):
    def __init__(self, ratio_threshold=0.7):
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        first_features: List[PointFeature],
        second_features: List[PointFeature],
        first_sensor_data: SensorData,
        second_sensor_data: SensorData,
    ):
        first_descs = np.array([feature.descriptor for feature in first_features])
        second_descs = np.array([feature.descriptor for feature in second_features])
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(first_descs, second_descs, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))
        return np.array(good_matches)
