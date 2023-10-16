import g2o
import numpy as np

from prime_slam.geometry.pose import Pose
from prime_slam.geometry.util import clip_lines
from prime_slam.slam.frame import Frame
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator
from prime_slam.projection.line_projector import LineProjector

__all__ = ["RGBDLinePoseEstimator"]


class RGBDLinePoseEstimator(PoseEstimator):
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
        self.lines_2d_shape = (-1, 2, 2)
        self.lines_3d_shape = (-1, 2, 3)
        self.projector = LineProjector()

    def estimate_absolute_pose(
        self, new_keyframe: Frame, map_lines_3d, matches, name
    ) -> Pose:
        new_lines_index = matches[:, 0]
        prev_lines_index = matches[:, 1]
        height, width = new_keyframe.sensor_measurement.depth.depth_map.shape[:2]

        new_lines = np.array(
            [
                keyline.coordinates
                for keyline in new_keyframe.observations.get_keyobjects(name)
            ]
        )
        new_lines = (
            clip_lines(new_lines, width=width, height=height)
            .astype(int)
            .reshape(self.lines_2d_shape)[new_lines_index]
        )
        map_lines_3d = map_lines_3d[prev_lines_index]

        nan_mask = np.logical_or.reduce(
            np.isinf(map_lines_3d) | np.isnan(map_lines_3d),
            axis=-1,
        )

        lines_obs = new_lines[~nan_mask]
        map_lines_3d = map_lines_3d[~nan_mask].reshape(self.lines_3d_shape)

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
                v1, measurement, information, map_lines_3d[i][0]
            )
            second_edge = self.__create_edge(
                v1, measurement, information, map_lines_3d[i][1]
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

        return Pose(v1.estimate().matrix())

    def estimate_relative_pose(
        self, new_keyframe: Frame, prev_keyframe: Frame, matches, name
    ) -> Pose:
        prev_lines = np.array(
            [
                keyline.coordinates
                for keyline in prev_keyframe.observations.get_keyobjects(name)
            ]
        )

        prev_depth_map = prev_keyframe.sensor_measurement.depth.depth_map
        prev_depth_scale = prev_keyframe.sensor_measurement.depth.depth_scale
        prev_intrinsics = prev_keyframe.sensor_measurement.depth.intrinsics
        height, width = prev_depth_map.shape[:2]

        prev_lines = clip_lines(prev_lines, width=width, height=height).astype(int)
        prev_lines_3d = self.projector.back_project(
            prev_lines, prev_depth_map, prev_depth_scale, prev_intrinsics, np.eye(4)
        )

        return self.estimate_absolute_pose(new_keyframe, prev_lines_3d, matches, name)

    @staticmethod
    def __create_optimizer() -> g2o.SparseOptimizer:
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
