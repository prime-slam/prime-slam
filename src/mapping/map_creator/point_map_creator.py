from src.mapping.point_map import PointMap
from src.mapping.map_creator.map_creator import MapCreator


class PointMapCreator(MapCreator):
    def create(self, projector):
        return PointMap(projector)
