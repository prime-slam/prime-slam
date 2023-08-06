import g2o
import numpy as np

from src.geometry.util import clip_lines
from src.keyframe import Keyframe
from src.relative_pose_estimation.estimator_base import PoseEstimatorBase
from src.sensor.depth import DepthImage


class RGBDLinePoseEstimator(PoseEstimatorBase):
    def __init__(
        self,
        camera_intrinsics,
        reprojection_threshold=20000,
        iterations_number=50,
        optimizer_iterations_number=30,
        edges_min_number=12,
    ):
        self.camera_intrinsics = camera_intrinsics
        self.iterations_number = iterations_number
        self.optimizer_iterations_number = optimizer_iterations_number
        self.reprojection_threshold = reprojection_threshold
        self.edges_min_number = edges_min_number
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]

    def estimate(self, new_keyframe: Keyframe, prev_keyframe: Keyframe, matches):
        new_lines = np.array(
            [
                np.array([feature.start_point, feature.end_point])
                for feature in new_keyframe.features
            ]
        ).reshape(-1, 4)
        prev_lines = np.array(
            [
                np.array([feature.start_point, feature.end_point])
                for feature in prev_keyframe.features
            ]
        ).reshape(-1, 4)

        prev_depth: DepthImage = prev_keyframe.sensor_measurement.depth
        height, width = prev_depth.depth_map.shape[:2]

        new_lines_index = matches[:, 0]
        prev_lines_index = matches[:, 1]
        lines_2d_shape = (-1, 2, 2)
        lines_3d_shape = (-1, 2, 3)
        new_lines = (
            clip_lines(new_lines, width=width, height=height)
            .astype(int)
            .reshape(lines_2d_shape)[new_lines_index]
        )
        prev_lines = (
            clip_lines(prev_lines, width=width, height=height)
            .astype(int)
            .reshape(lines_2d_shape)[prev_lines_index]
        )
        prev_lines_3d = prev_depth.back_project_lines(prev_lines)

        nan_mask = np.logical_or.reduce(
            np.isinf(prev_lines_3d) | np.isnan(prev_lines_3d),
            axis=-1,
        )

        lines_obs = new_lines[~nan_mask]
        keyframe_3d_lines = prev_lines_3d[~nan_mask].reshape(lines_3d_shape)

        optimizer = self.__create_optimizer()
        v1 = g2o.VertexSE3Expmap()
        v1.set_id(0)
        v1.set_fixed(False)
        optimizer.add_vertex(v1)
        line_edges = []
        v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))

        lines_number = len(lines_obs)
        information = np.eye(3)

        for i in range(lines_number):
            measurement = np.cross(
                np.append(lines_obs[i][0], 1), np.append(lines_obs[i][1], 1)
            )
            first_edge = self.__create_edge(
                v1, measurement, information, keyframe_3d_lines[i][0]
            )
            second_edge = self.__create_edge(
                v1, measurement, information, keyframe_3d_lines[i][1]
            )
            optimizer.add_edge(first_edge)
            optimizer.add_edge(second_edge)
            line_edges.append((first_edge, second_edge))

        inlier_mask = np.ones(lines_number, dtype=bool)

        for i in range(self.iterations_number):
            v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))
            optimizer.initialize_optimization()
            optimizer.optimize(self.optimizer_iterations_number)

            for j, (first_edge, second_edge) in enumerate(line_edges):
                if (
                    first_edge.chi2() + second_edge.chi2()
                ) / 2 > self.reprojection_threshold / (i + 1) ** 2:
                    inlier_mask[j] = False
                    first_edge.set_level(1)
                    second_edge.set_level(1)
                else:
                    inlier_mask[j] = True
                    first_edge.set_level(0)
                    second_edge.set_level(0)

                if i == self.iterations_number - 2:
                    first_edge.set_robust_kernel(None)
                    second_edge.set_robust_kernel(None)

            if 2 * np.count_nonzero(inlier_mask) < self.edges_min_number:
                break

        return v1.estimate().matrix()

    @staticmethod
    def __create_optimizer():
        optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)

        return optimizer

    def __create_edge(self, v1, measurement, information, Xw):
        edge = g2o.EdgeLineProjectXYZOnlyPose()
        edge.set_vertex(0, v1)
        edge.set_measurement(measurement)
        edge.set_information(information)
        edge.set_robust_kernel(g2o.RobustKernelHuber())
        edge.fx = self.fx
        edge.fy = self.fy
        edge.cx = self.cx
        edge.cy = self.cy
        edge.Xw = Xw

        return edge
