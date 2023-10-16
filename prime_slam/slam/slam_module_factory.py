from typing import List

from prime_slam.slam.config.mapping_config import MappingConfig
from prime_slam.slam.config.observation_config import ObservationConfig
from prime_slam.slam.config.slam_config import SLAMConfig
from prime_slam.slam.config.tracking_config import TrackingConfig
from prime_slam.slam.mapping.mapping import Mapping
from prime_slam.slam.observation_creator import ObservationCreator
from prime_slam.slam.tracking.tracker import Tracker


class SLAMModuleFactory:
    def __init__(self, configs: List[SLAMConfig]):
        self._configs = configs

    def create_tracker(self) -> Tracker:
        return Tracker(
            [
                TrackingConfig.from_slam_config(slam_config)
                for slam_config in self._configs
            ]
        )

    def create_mapping(self) -> Mapping:
        return Mapping(
            [
                MappingConfig.from_slam_config(slam_config)
                for slam_config in self._configs
            ]
        )

    def create_observation_creator(self) -> ObservationCreator:
        return ObservationCreator(
            [
                ObservationConfig.from_slam_config(slam_config)
                for slam_config in self._configs
            ]
        )
