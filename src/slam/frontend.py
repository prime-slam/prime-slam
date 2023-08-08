from typing import List

from src.detection.detector_base import Detector
from src.keyframe import Keyframe
from src.keyframe_selection.keyframe_selector import KeyframeSelector
from src.matching.matcher_base import Matcher
from src.relative_pose_estimation.estimator_base import PoseEstimatorBase


class PrimeSLAMFrontend:
    def __init__(
        self,
        observation_detector: Detector,
        observation_matcher: Matcher,
        relative_pose_estimator: PoseEstimatorBase,
        keyframe_selector: KeyframeSelector,
        init_pose,
    ):
        self.observation_detector = observation_detector
        self.observation_matcher = observation_matcher
        self.relative_pose_estimator = relative_pose_estimator
        self.keyframes: List[Keyframe] = []
        self.keyframe_selector = keyframe_selector
        self.landmarks = []
        self.init_pose = init_pose

    def initialize_map(self, keyframe):
        keyframe.update_pose(self.init_pose)
        self.keyframes.append(keyframe)

    def process_frame(self, sensor_data):
        observations = self.observation_detector.detect(sensor_data)
        new_keyframe = Keyframe(observations, sensor_data)
        if len(self.keyframes) == 0:
            self.initialize_map(new_keyframe)
            return

        last_keyframe = self.keyframes[-1]
        matches = self.observation_matcher.match(
            observations,
            last_keyframe.observations,
            sensor_data,
            last_keyframe.sensor_measurement,
        )
        relative_pose = self.relative_pose_estimator.estimate(
            new_keyframe, last_keyframe, matches
        )
        absolute_pose = self.__calculate_new_absolute_pose(
            last_keyframe.world_to_camera_transform, relative_pose
        )
        new_keyframe.update_pose(absolute_pose)
        if self.keyframe_selector.is_selected(new_keyframe):
            self.keyframes.append(new_keyframe)

    @staticmethod
    def __calculate_new_absolute_pose(prev_abs_pose, relative_pose):
        return relative_pose @ prev_abs_pose
