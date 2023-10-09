from src.geometry.pose import Pose
from src.observation.observations_batch import ObservationsBatch
from src.sensor.sensor_data import SensorData

__all__ = ["Frame"]


class Frame:
    def __init__(
        self,
        observations: ObservationsBatch,
        sensor_measurement: SensorData,
        local_map=None,  # TODO: make abstract map
        world_to_camera_transform: Pose = None,
        is_keyframe=False,
        identifier: int = 0,
    ):
        self.observations: ObservationsBatch = observations
        self.sensor_measurement = sensor_measurement
        self.local_map = local_map
        self._world_to_camera_transform: Pose = world_to_camera_transform
        self._camera_to_world_transform: Pose = (
            world_to_camera_transform.inverse()
            if world_to_camera_transform is not None
            else None
        )
        self._is_keyframe = is_keyframe
        self.identifier = identifier

    def __hash__(self):
        return self.identifier

    @property
    def is_keyframe(self):
        return self._is_keyframe

    @is_keyframe.setter
    def is_keyframe(self, value):
        self._is_keyframe = value

    def update_pose(self, new_pose: Pose):
        self._world_to_camera_transform = new_pose
        self._camera_to_world_transform = new_pose.inverse()

    @property
    def world_to_camera_transform(self):
        return self._world_to_camera_transform.transformation

    @property
    def world_to_camera_rotation(self):
        return self._world_to_camera_transform.rotation

    @property
    def camera_to_world_rotation(self):
        return self._camera_to_world_transform.rotation

    @property
    def world_to_camera_translation(self):
        return self._world_to_camera_transform.translation

    @property
    def camera_to_world_translation(self):
        return self._camera_to_world_transform.translation

    @property
    def origin(self):
        return -(self.world_to_camera_rotation @ self.camera_to_world_translation)
