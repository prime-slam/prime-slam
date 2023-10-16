from abc import ABC, abstractmethod


class MapCreator(ABC):
    def __init__(self, projector, landmark_name: str):
        self.projector = projector
        self.landmark_name: str = landmark_name

    @abstractmethod
    def create(self):
        pass
