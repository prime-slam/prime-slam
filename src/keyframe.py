from src.observation.observations_batch import ObservationsBatch
from src.sensor.sensor_data_base import SensorData


class Keyframe:
    def __init__(
        self,
        observations: ObservationsBatch,
        sensor_measurement: SensorData,
        world_to_camera_transform=None,
    ):
        self.observations = observations
        self.sensor_measurement = sensor_measurement
        self.world_to_camera_transform = world_to_camera_transform

    def update_pose(self, new_pose):
        self.world_to_camera_transform = new_pose
