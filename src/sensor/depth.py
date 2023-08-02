import numpy as np

from src.sensor.sensor_data_base import SensorData


class DepthImage(SensorData):
    def __init__(
        self,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        depth_scale: float,
    ):
        self.depth_map = depth_map
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale

    def back_project_points(self, points_2d):
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        x, y = points_2d[:, 0].astype(int), points_2d[:, 1].astype(int)
        depths = self.depth_map[y, x]
        nonzero_depths = depths != 0
        z_3d = depths / self.depth_scale
        x_3d = np.zeros(len(z_3d))
        y_3d = np.zeros(len(z_3d))
        x_3d[nonzero_depths] = (x[nonzero_depths] - cx) / fx * z_3d[nonzero_depths]
        y_3d[nonzero_depths] = (y[nonzero_depths] - cy) / fy * z_3d[nonzero_depths]
        # set np.nan if depth is zero
        z_3d[~nonzero_depths] = np.nan
        x_3d[~nonzero_depths] = np.nan
        y_3d[~nonzero_depths] = np.nan

        return np.column_stack([x_3d, y_3d, z_3d])
