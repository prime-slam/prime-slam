# from src.mapping.map import Map
from src.observation.observations_batch import ObservationsBatch
from src.sensor.sensor_data_base import SensorData


class Frame:
    def __init__(
        self,
        observations: ObservationsBatch,
        sensor_measurement: SensorData,
        local_map=None,  # TODO: make abstract map
        world_to_camera_transform=None,
        is_keyframe=False,
        identifier: int = 0,
    ):
        self.observations: ObservationsBatch = observations
        self.sensor_measurement = sensor_measurement
        self.local_map = local_map
        self.world_to_camera_transform = None
        self.camera_to_world_transform = None
        self._world_to_camera_rotation = None
        self._world_to_camera_translation = None
        self._camera_to_world_rotation = None
        self._camera_to_world_translation = None
        self._is_keyframe = is_keyframe
        if world_to_camera_transform is not None:
            self.update_pose(world_to_camera_transform)
        self.identifier = identifier

    def __hash__(self):
        return self.identifier

    @property
    def is_keyframe(self):
        return self._is_keyframe

    @is_keyframe.setter
    def is_keyframe(self, value):
        self._is_keyframe = value

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
