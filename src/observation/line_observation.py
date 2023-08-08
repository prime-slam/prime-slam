from src.observation.observation import Observation


class LineObservation(Observation):
    def __init__(self, start_point, end_point, descriptor=None):
        self.start_point = start_point
        self.end_point = end_point
        self.descriptor = descriptor
