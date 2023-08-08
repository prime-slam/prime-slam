from src.observation.observation import Observation


class PointObservation(Observation):
    def __init__(self, x, y, descriptor=None):
        self.x = x
        self.y = y
        self.descriptor = descriptor
