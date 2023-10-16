import g2o
import numpy as np

from prime_slam.slam.frame.frame import Frame
from prime_slam.geometry.pose import Pose
from prime_slam.slam.tracking.pose_estimation.estimator import PoseEstimator
from prime_slam.projection.point_projection import PointProjector

__all__ = ["RGBDPointPoseEstimator"]


class RGBDPointPoseEstimator(PoseEstimator):
    def __init__(
        self,
        camera_intrinsics,
        reprojection_threshold=30,
        iterations_number=4,
        optimizer_iterations_number=10,
        edges_min_number=20,
    ):
        self.reprojection_threshold = reprojection_threshold
        self.iterations_number = iterations_number
        self.edges_min_number = edges_min_number
        self.optimizer_iterations_number = optimizer_iterations_number
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]
        self.projector = PointProjector()

    def estimate_absolute_pose(
        self, new_keyframe: Frame, map_3d_points, matches, name
    ) -> Pose:
        new_keypoints = np.array(
            [
                keypoint.coordinates
                for keypoint in new_keyframe.observations.get_keyobjects(name)
            ]
        )
        new_keypoints_index = matches[:, 0]
        prev_keypoints_index = matches[:, 1]

        kpts_obs = new_keypoints[new_keypoints_index]
        map_3d_points = map_3d_points[prev_keypoints_index]

        nan_mask = np.logical_or.reduce(
            np.isinf(map_3d_points) | np.isnan(map_3d_points),
            axis=-1,
        )

        kpts_obs = kpts_obs[~nan_mask]
        map_3d_points = map_3d_points[~nan_mask]

        edges = []
        optimizer = self.__create_optimizer()

        v1 = g2o.VertexSE3Expmap()
        v1.set_id(0)
        v1.set_fixed(False)
        optimizer.add_vertex(v1)

        v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))

        for i in range(len(kpts_obs)):
            edge = g2o.EdgeSE3ProjectXYZOnlyPose()

            edge.set_vertex(0, v1)
            edge.set_measurement(kpts_obs[i][:2])
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.fx = self.fx
            edge.fy = self.fy
            edge.cx = self.cx
            edge.cy = self.cy
            edge.Xw = map_3d_points[i]

            optimizer.add_edge(edge)
            edges.append(edge)

        inl_mask = np.ones((len(kpts_obs),), dtype=bool)

        for i in range(self.iterations_number):
            v1.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros((3,))))

            optimizer.initialize_optimization()
            optimizer.optimize(self.optimizer_iterations_number)

            for j, edge in enumerate(edges):
                if edge.chi2() > self.reprojection_threshold / (i + 1) ** 2:
                    inl_mask[j] = False
                    edge.set_level(1)
                else:
                    inl_mask[j] = True
                    edge.set_level(0)

                if i == self.iterations_number - 2:
                    edge.set_robust_kernel(None)

            if np.sum(inl_mask) < self.edges_min_number:
                break

        return Pose(v1.estimate().matrix())

    def estimate_relative_pose(
        self, new_keyframe: Frame, prev_keyframe: Frame, matches, name
    ) -> Pose:
        prev_keypoints = np.array(
            [
                keypoint.coordinates
                for keypoint in prev_keyframe.observations.get_keyobjects(name)
            ]
        )
        prev_depth_map = prev_keyframe.sensor_measurement.depth.depth_map
        prev_depth_scale = prev_keyframe.sensor_measurement.depth.depth_scale
        prev_intrinsics = prev_keyframe.sensor_measurement.depth.intrinsics
        prev_keypoints_3d = self.projector.back_project(
            prev_keypoints, prev_depth_map, prev_depth_scale, prev_intrinsics, np.eye(4)
        )

        return self.estimate_absolute_pose(
            new_keyframe, prev_keypoints_3d, matches, name
        )

    @staticmethod
    def __create_optimizer():
        optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)

        return optimizer
