class Keyframe:
    def __init__(self, features, sensor_measurement, world_to_camera_transform=None):
        self.observations = features
        self.sensor_measurement = sensor_measurement
        self.world_to_camera_transform = world_to_camera_transform

    def update_pose(self, new_pose):
        self.world_to_camera_transform = new_pose
