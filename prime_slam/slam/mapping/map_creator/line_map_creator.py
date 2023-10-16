from prime_slam.slam.mapping.line_map import LineMap
from prime_slam.slam.mapping.map_creator.map_creator import MapCreator


class LineMapCreator(MapCreator):
    def __init__(self, projector, landmark_name: str):
        super().__init__(projector, landmark_name)

    def create(self):
        return LineMap(self.projector, self.landmark_name)
