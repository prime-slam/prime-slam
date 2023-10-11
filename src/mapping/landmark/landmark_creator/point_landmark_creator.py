from src.mapping.landmark.landmark_creator.landmark_creator import LandmarkCreator
from src.mapping.landmark.point_landmark import PointLandmark


class PointLandmarkCreator(LandmarkCreator):
    def create(self, current_id, landmark_position, descriptor, frame):
        return PointLandmark(current_id, landmark_position, descriptor, frame)
