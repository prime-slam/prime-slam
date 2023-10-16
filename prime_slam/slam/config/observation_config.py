from dataclasses import dataclass

from prime_slam.observation.description.descriptor import Descriptor
from prime_slam.observation.detection.detector import Detector
from prime_slam.observation.filter.observation_filter import ObservationsFilter

__all__ = ["ObservationConfig"]

from prime_slam.slam.config.slam_config import SLAMConfig


@dataclass
class ObservationConfig:
    detector: Detector
    descriptor: Descriptor
    observations_filter: ObservationsFilter
    observation_name: str

    @classmethod
    def from_slam_config(cls, slam_config: SLAMConfig):
        return cls(
            detector=slam_config.detector,
            descriptor=slam_config.descriptor,
            observations_filter=slam_config.observations_filter,
            observation_name=slam_config.observation_name,
        )
