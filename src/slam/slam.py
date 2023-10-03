from typing import List

from src.frame import Frame
from src.observation.observation_creator import ObservationsCreator
from src.slam.backend import Backend
from src.slam.frontend import Frontend


# TODO: add id to landmarks
# TODO: add best descriptor choice
# TODO: viewing direction does not work
class PrimeSLAM:
    def __init__(
        self,
        observation_creators: List[ObservationsCreator],
        frontend: Frontend,
        backend: Backend,
        init_pose,
    ):
        self.observation_creators = observation_creators
        self.keyframes: List[Frame] = []
        self.backend = backend
        self.frontend = frontend
        self.init_pose = init_pose

    def process_frame(self, sensor_data):
        new_frame = self.frontend.process_sensor_data(sensor_data)
        if len(self.keyframes) == 0:
            # TODO: fail if frame contains zero depth point
            self.__initialize(new_frame)
            return
        last_keyframe = self.keyframes[-1]
        new_frame = self.frontend.track(last_keyframe, new_frame, sensor_data)

        if new_frame.is_keyframe:
            self.keyframes.append(new_frame)
            for observation_name in new_frame.observations.observation_names:
                self.optimize_graph(observation_name)

    def __initialize(self, frame):
        frame.update_pose(self.init_pose)
        self.keyframes.append(frame)
        self.frontend.initialize_tracking(frame)

    def optimize_graph(self, landmark_name):
        new_poses, new_landmark_positions = self.backend.optimize(self.frontend.graph)
        self.frontend.mapping.map.update_position(new_landmark_positions, landmark_name)
        for kf, new_pose in zip(self.keyframes, new_poses):
            kf.update_pose(new_pose)
        self.frontend.graph.update_poses(new_poses)
        self.frontend.graph.update_landmarks_positions(new_landmark_positions)
        self.frontend.mapping.map.recalculate_mean_viewing_directions(landmark_name)
