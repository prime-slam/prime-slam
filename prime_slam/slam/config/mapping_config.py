from dataclasses import dataclass

from prime_slam.slam.config.slam_config import SLAMConfig
from prime_slam.slam.mapping.map_creator.map_creator import MapCreator
from prime_slam.projection.projector import Projector

__all__ = ["MappingConfig"]


@dataclass
class MappingConfig:
    projector: Projector
    map_creator: MapCreator
    observation_name: str

    @classmethod
    def from_slam_config(cls, slam_config: SLAMConfig):
        return cls(
            projector=slam_config.projector,
            map_creator=slam_config.map_creator,
            observation_name=slam_config.observation_name,
        )
