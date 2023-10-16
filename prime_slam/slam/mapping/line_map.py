from typing import List

from prime_slam.slam.frame.frame import Frame
from prime_slam.slam.mapping.landmark.landmark import Landmark
from prime_slam.slam.mapping.landmark.line_landmark import LineLandmark
from prime_slam.slam.mapping.map import Map
from prime_slam.projection.projector import Projector


class LineMap(Map):
    def __init__(
        self, projector: Projector, landmark_name, landmarks: List[Landmark] = None
    ):
        super().__init__(projector, landmark_name, landmarks)

    def get_visible_map(
        self,
        frame: Frame,
    ) -> Map:
        raise NotImplementedError()

    def create_landmark(self, current_id, landmark_position, descriptor, frame):
        return LineLandmark(current_id, landmark_position, descriptor, frame)
