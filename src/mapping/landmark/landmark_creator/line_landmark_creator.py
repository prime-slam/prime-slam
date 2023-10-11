from src.mapping.landmark.landmark_creator.landmark_creator import LandmarkCreator
from src.mapping.landmark.line_landmark import LineLandmark


class LineLandmarkCreator(LandmarkCreator):
    def create(self, current_id, landmark_position, descriptor, frame):
        return LineLandmark(current_id, landmark_position, descriptor, frame)
