from src.feature.feature import Feature


class LineFeature(Feature):
    def __init__(self, start_point, end_point, descriptor=None):
        self.start_point = start_point
        self.end_point = end_point
        self.descriptor = descriptor
