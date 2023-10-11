from typing import Dict

from src.frame import Frame
from src.mapping.landmark.landmark import Landmark
from src.mapping.landmark.landmark_creator.line_landmark_creator import LineLandmarkCreator
from src.mapping.map import Map
from src.projection import Projector


class LineMap(Map):
    def __init__(self, projector: Projector, landmarks: Dict[int, Landmark] = None):
        super().__init__(projector, LineLandmarkCreator(), landmarks)

    def get_visible_map(
        self,
        frame: Frame,
    ):
        raise NotImplementedError()
