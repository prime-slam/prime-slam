import numpy as np


class Landmark:
    def __init__(self, position, feature_descriptor, viewing_direction):
        self.position = position
        self.feature_descriptor = feature_descriptor
        self._mean_viewing_direction = viewing_direction / np.linalg.norm(
            viewing_direction
        )

    def add_viewing_direction(self, viewing_direction):
        viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)
        self._mean_viewing_direction += viewing_direction
        self._mean_viewing_direction /= np.linalg.norm(self._mean_viewing_direction)

    @property
    def mean_viewing_direction(self):
        return self._mean_viewing_direction
