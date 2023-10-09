from abc import ABC

__all__ = ["Node"]


class Node(ABC):
    def __init__(self, identifier):
        self.identifier = identifier

    def __hash__(self):
        return self.identifier
