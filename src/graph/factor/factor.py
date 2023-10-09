from abc import ABC

__all__ = ["Factor"]


class Factor(ABC):
    def __init__(self, from_node, to_node, information):
        self.from_node = from_node
        self.to_node = to_node
        self.information = information
