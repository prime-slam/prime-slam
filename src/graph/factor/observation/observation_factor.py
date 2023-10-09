import numpy as np

from src.graph.factor.factor import Factor
from src.sensor.rgbd import RGBDImage

__all__ = ["ObservationFactor"]


class ObservationFactor(Factor):
    def __init__(
        self,
        keyobject_position_2d,
        pose_node,
        landmark_node,
        sensor_measurement: RGBDImage,
        information=None,
    ):
        super().__init__(
            pose_node,
            landmark_node,
            information if information is not None else np.eye(2),
        )
        self._keyobject_2d = keyobject_position_2d
        self.depth_map = sensor_measurement.depth.depth_map
        self.bf = sensor_measurement.bf
        self.stereo_coords = self.__convert_to_stereo(
            keyobject_position_2d,
            self.depth_map / sensor_measurement.depth.depth_scale,
            sensor_measurement.bf,
        )

    @property
    def observation(self):
        return self.stereo_coords

    @staticmethod
    def __convert_to_stereo(coords, depth_map: np.ndarray, bf):
        x, y = coords
        d = depth_map[int(y), int(x)]
        return np.array([x, y, x - bf / d])
