from abc import ABC, abstractmethod


class MapCreator(ABC):
    @abstractmethod
    def create(self, projector):
        pass
