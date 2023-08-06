import numpy as np
import open3d as o3d

from pathlib import Path
from skimage import io

from src.detection.lines.lsd import LSD
from src.detection.points.orb import ORB
from src.detection.points.sift import SIFT
from src.geometry.util import clip_lines
from src.keyframe_selection.every_nth_keyframe_selector import EveryNthKeyframeSelector
from src.matching.lines.lbd import LBD
from src.matching.points.orb_matcher import ORBMatcher
from src.matching.points.sift_matcher import SIFTMatcher
from src.relative_pose_estimation.rgbd_line_pose_estimator import RGBDLinePoseEstimator
from src.geometry.transform import make_homogeneous_matrix
from src.relative_pose_estimation.rgbd_point_pose_estimator import (
    RGBDPointPoseEstimator,
)

from src.sensor.depth import DepthImage
from src.sensor.rgb import RGBImage
from src.sensor.rgbd import RGBDImage
from src.slam.frontend import PrimeSLAMFrontend


def create_point_map(keyframes):
    abs_poses = [kf.world_to_camera_transform for kf in keyframes]
    depths = [kf.sensor_measurement.depth for kf in keyframes]
    features_batch = [kf.features for kf in keyframes]

    keypoints_batch = [
        np.array([np.array([feature.x, feature.y]) for feature in features])
        for features in features_batch
    ]

    keypoints_3d_batch = [
        depth.back_project_points(keypoints)
        for keypoints, depth in zip(keypoints_batch, depths)
    ]

    return [
        get_point_cloud(keypoints_3D).transform(np.linalg.inv(pose))
        for keypoints_3D, pose in zip(keypoints_3d_batch, abs_poses)
    ]


def create_line_map(keyframes):
    abs_poses = [kf.world_to_camera_transform for kf in keyframes]
    depths = [kf.sensor_measurement.depth for kf in keyframes]
    features_batch = [kf.features for kf in keyframes]

    lines_batch = [
        np.array(
            [np.array([feature.start_point, feature.end_point]) for feature in features]
        ).reshape(-1, 4)
        for features in features_batch
    ]
    lines_2d_shape = (-1, 2, 2)
    lines_batch = [
        clip_lines(
            lines, height=depth.depth_map.shape[0], width=depth.depth_map.shape[1]
        )
        .astype(int)
        .reshape(lines_2d_shape)
        for lines, depth in zip(lines_batch, depths)
    ]

    lines_3d_batch = [
        depth.back_project_lines(keypoints).reshape(-1, 2, 3)
        for keypoints, depth in zip(lines_batch, depths)
    ]
    return [
        get_line_set(lines_3D).transform(np.linalg.inv(pose))
        for lines_3D, pose in zip(lines_3d_batch, abs_poses)
    ]


def get_point_cloud(points_3d):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    return point_cloud


def get_line_set(lines_3d):
    line_set = o3d.geometry.LineSet()
    points = []
    lines = []

    for i, edges in enumerate(lines_3d):
        points.append(edges[0])
        points.append(edges[1])
        lines.append((2 * i, 2 * i + 1))

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 1, 0])

    return line_set


if __name__ == "__main__":
    images_path = Path("/home/jeten/Desktop/evolin/Predictions/lr_kt2/rgb/")
    depth_path = Path("/home/jeten/Desktop/evolin/Predictions/lr_kt2/depth/")
    intrinsics_path = Path(
        "/home/jeten/Desktop/evolin/Predictions/lr_kt2/calibration_matrix.txt"
    )

    images_paths = sorted(images_path.iterdir())
    depth_paths = sorted(depth_path.iterdir())
    intrinsics = make_homogeneous_matrix(np.genfromtxt(intrinsics_path))
    depth_scaler = 5000
    images_number = len(depth_paths)

    features = "ORB"

    if features == "ORB":
        extractor = ORB(nfeatures=1000)
        matcher = ORBMatcher()
        relative_pose_estimator = RGBDPointPoseEstimator(intrinsics)
        create_map = create_point_map
    elif features == "SIFT":
        extractor = SIFT()
        matcher = SIFTMatcher()
        relative_pose_estimator = RGBDPointPoseEstimator(intrinsics)
        create_map = create_point_map
    elif features == "LSD":
        extractor = LSD()
        matcher = LBD()
        relative_pose_estimator = RGBDLinePoseEstimator(intrinsics)
        create_map = create_line_map
    else:
        raise ValueError(f"Unknown features: {features}")

    images = [io.imread(path) for path in images_paths]
    depths = [io.imread(path) for path in depth_paths]

    frames = [
        RGBDImage(RGBImage(img), DepthImage(depth, intrinsics, depth_scaler))
        for img, depth in zip(images, depths)
    ]
    keyframe_selector = EveryNthKeyframeSelector(n=10)

    slam = PrimeSLAMFrontend(
        extractor,
        matcher,
        relative_pose_estimator,
        keyframe_selector,
        init_pose=np.eye(4),
    )

    for frame in frames:
        slam.process_frame(frame)

    o3d.visualization.draw_geometries(create_map(slam.keyframes))
