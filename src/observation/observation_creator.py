from itertools import compress

import numpy as np

from typing import Callable, Type

from src.mapping.landmark.landmark_creator.landmark_creator import LandmarkCreator
from src.observation.description import Descriptor
from src.observation.detection.detector import Detector
from src.observation.mask.coordinates_mask import CoordinatesMask
from src.observation.observation import Observation
from src.pose_estimation.estimator import PoseEstimator
from src.projection.projector import Projector
from src.sensor.sensor_data import SensorData

__all__ = ["ObservationsCreator"]


class ObservationsCreator:  # TODO: change name
    def __init__(
        self,
        detector: Detector,
        descriptor: Descriptor,
        matcher: Callable[[np.ndarray, np.ndarray], np.ndarray],
        projector: Projector,
        pose_estimator: PoseEstimator,
        coordinates_mask: CoordinatesMask,
        observation_name: str,
        landmark_creator: LandmarkCreator,
    ):
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.projector = projector
        self.pose_estimator = pose_estimator
        self.coordinates_mask = coordinates_mask
        self._observation_name = observation_name
        self.landmark_creator = landmark_creator

    def create_observations(self, sensor_data: SensorData):
        keyobjects = self.detector.detect(sensor_data)
        descriptors = self.descriptor.descript(keyobjects, sensor_data)
        coordinates = np.array([keyobject.coordinates for keyobject in keyobjects])
        observations = [
            Observation(keyobject, descriptor)
            for keyobject, descriptor in zip(keyobjects, descriptors)
        ]

        mask = self.coordinates_mask.create(coordinates, sensor_data)

        return list(compress(observations, mask))

    def create_landmark(self, current_id, landmark_position, descriptor, frame):
        return self.landmark_creator.create(
            current_id, landmark_position, descriptor, frame
        )

    @property
    def observation_name(self):
        return self._observation_name
