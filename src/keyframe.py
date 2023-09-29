from src.observation.observations_batch import ObservationsBatch
from src.sensor.sensor_data_base import SensorData


class Keyframe:
    def __init__(
        self,
        observations: ObservationsBatch,
        sensor_measurement: SensorData,
        world_to_camera_transform=None,
        identifier: int = 0,
    ):
        self.observations: ObservationsBatch = observations
        self.sensor_measurement = sensor_measurement
        self.world_to_camera_transform = None
        self.camera_to_world_transform = None
        self._world_to_camera_rotation = None
        self._world_to_camera_translation = None
        self._camera_to_world_rotation = None
        self._camera_to_world_translation = None
        if world_to_camera_transform is not None:
            self.update_pose(world_to_camera_transform)
        self.identifier = identifier

    def __hash__(self):
        return self.identifier

    def update_pose(self, new_pose):
        self.world_to_camera_transform = new_pose
        self.camera_to_world_transform = new_pose.T
        self._world_to_camera_rotation = self.world_to_camera_transform[:3, :3]
        self._world_to_camera_translation = self.world_to_camera_transform[:3, 3]
        self._camera_to_world_rotation = self.camera_to_world_transform[:3, :3]
        self._camera_to_world_translation = self.camera_to_world_transform[:3, 3]

    @property
    def world_to_camera_rotation(self):
        return self._world_to_camera_rotation

    @property
    def camera_to_world_rotation(self):
        return self._camera_to_world_rotation

    @property
    def world_to_camera_translation(self):
        return self._world_to_camera_translation

    @property
    def camera_to_world_translation(self):
        return self._camera_to_world_translation

    @property
    def origin(self):
        return -(self.world_to_camera_rotation @ self.camera_to_world_translation)
