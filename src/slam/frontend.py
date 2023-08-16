from typing import List, Callable

import numpy as np

from src.description.descriptor_base import Descriptor
from src.detection.detector_base import Detector
from src.keyframe import Keyframe
from src.keyframe_selection.keyframe_selector import KeyframeSelector
from src.observation.observation_creator import ObservationsCreator
from src.relative_pose_estimation.estimator_base import PoseEstimatorBase


class PrimeSLAMFrontend:
    def __init__(
        self,
        detector: Detector,
        descriptor: Descriptor,
        matcher: Callable[[np.ndarray, np.ndarray], np.ndarray],
        estimator: PoseEstimatorBase,
        keyframe_selector: KeyframeSelector,
        init_pose,
    ):
        self.observation_creator = ObservationsCreator(detector, descriptor)
        self.descriptor_matcher = matcher
        self.relative_pose_estimator = estimator
        self.keyframes: List[Keyframe] = []
        self.keyframe_selector = keyframe_selector
        self.landmarks = []
        self.init_pose = init_pose

    def initialize_map(self, keyframe):
        keyframe.update_pose(self.init_pose)
        self.keyframes.append(keyframe)

    def process_frame(self, sensor_data):
        observations = self.observation_creator.create_observations(sensor_data)
        new_keyframe = Keyframe(observations, sensor_data)
        if len(self.keyframes) == 0:
            self.initialize_map(new_keyframe)
            return

        last_keyframe = self.keyframes[-1]
        matches = self.descriptor_matcher(
            observations.descriptors,
            last_keyframe.observations.descriptors,
        )
        relative_pose = self.relative_pose_estimator.estimate(
            new_keyframe, last_keyframe, matches
        )
        absolute_pose = relative_pose @ last_keyframe.world_to_camera_transform

        new_keyframe.update_pose(absolute_pose)
        if self.keyframe_selector.is_selected(new_keyframe):
            self.keyframes.append(new_keyframe)
