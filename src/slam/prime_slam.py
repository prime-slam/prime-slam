from src.slam.backend import Backend
from src.slam.frontend import Frontend
from src.slam.slam import SLAM


class PrimeSLAM(SLAM):
    def __init__(
        self,
        frontend: Frontend,
        backend: Backend,
    ):
        self.backend = backend
        self.frontend = frontend

    @property
    def trajectory(self):
        return self.frontend.trajectory

    @property
    def map(self):
        return self.frontend.map

    def process_sensor_data(self, sensor_data):
        new_frame = self.frontend.process_sensor_data(sensor_data)
        if new_frame.is_keyframe:
            for observation_name in new_frame.observations.observation_names:
                self.__optimize_graph(observation_name)

    def __optimize_graph(self, landmark_name):
        new_poses, new_landmark_positions = self.backend.optimize(self.frontend.graph)
        self.frontend.update_poses(new_poses)
        self.frontend.update_landmark_positions(new_landmark_positions, landmark_name)
