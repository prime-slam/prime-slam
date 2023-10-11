from src.mapping.line_map import LineMap
from src.mapping.map_creator.map_creator import MapCreator


class LineMapCreator(MapCreator):
    def create(self, projector):
        return LineMap(projector)
