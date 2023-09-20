from abc import ABC


class Factor(ABC):
    def __init__(self, from_node, to_node, information):
        self.from_node = from_node
        self.to_node = to_node
        self.information = information
