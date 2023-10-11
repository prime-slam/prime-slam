from abc import ABC, abstractmethod


class LandmarkCreator(ABC):
    @abstractmethod
    def create(self, current_id, landmark_position, descriptor, frame):
        pass
