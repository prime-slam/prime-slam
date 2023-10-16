from prime_slam.slam.mapping.point_map import PointMap
from prime_slam.slam.mapping.map_creator.map_creator import MapCreator


class PointMapCreator(MapCreator):
    def __init__(self, projector, landmark_name: str):
        super().__init__(projector, landmark_name)

    def create(self):
        return PointMap(self.projector, self.landmark_name)
