from src.feature.feature import Feature


class PointFeature(Feature):
    def __init__(self, x, y, descriptor=None):
        self.x = x
        self.y = y
        self.descriptor = descriptor
