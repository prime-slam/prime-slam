from typing import List

from prime_slam.observation.observation import Observation
from prime_slam.observation.observations_batch import ObservationsBatch
from prime_slam.sensor.sensor_data import SensorData
from prime_slam.slam.config.observation_config import ObservationConfig


class ObservationCreator:
    def __init__(self, observation_configs: List[ObservationConfig]):
        self.observation_configs = observation_configs

    def create_observations(self, sensor_data: SensorData) -> ObservationsBatch:
        observation_batch = []
        names = []
        for config in self.observation_configs:
            keyobjects = config.detector.detect(sensor_data)
            descriptors = config.descriptor.descript(keyobjects, sensor_data)
            observations = [
                Observation(keyobject, descriptor)
                for keyobject, descriptor in zip(keyobjects, descriptors)
            ]

            observations = config.observations_filter.apply(observations, sensor_data)
            observation_batch.append(observations)
            names.append(config.observation_name)

        observations_batch = ObservationsBatch(observation_batch, names)

        return observations_batch
