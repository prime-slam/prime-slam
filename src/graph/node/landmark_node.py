from src.graph.node.node_base import Node


class LandmarkNode(Node):
    def __init__(self, identifier, position=None):
        super().__init__(identifier)
        self._position = position

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._position = new_position
