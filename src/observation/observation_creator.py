import numpy as np

from typing import Callable

from src.observation.description import Descriptor
from src.observation.detection.detector import Detector
from src.pose_estimation.estimator import PoseEstimator
from src.projection.projector import Projector
from src.sensor.sensor_data import SensorData

__all__ = ["ObservationsCreator"]


class ObservationsCreator:
    def __init__(
        self,
        detector: Detector,
        descriptor: Descriptor,
        matcher: Callable[[np.ndarray, np.ndarray], np.ndarray],
        projector: Projector,
        pose_estimator: PoseEstimator,
        observation_name: str,
    ):
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.projector = projector
        self.pose_estimator = pose_estimator
        self._observation_name = observation_name

    def create_observations(self, sensor_data: SensorData):
        keyobjects = self.detector.detect(sensor_data)
        descriptors = self.descriptor.descript(keyobjects, sensor_data)

        return keyobjects, descriptors

    @property
    def observation_name(self):
        return self._observation_name
