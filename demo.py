import argparse
import numpy as np
import open3d as o3d

from functools import partial
from pathlib import Path
from skimage import io
from skimage.feature import match_descriptors


from src.geometry.io import read_poses
from src.geometry.pose import Pose
from src.mapping.landmark.landmark_creator.point_landmark_creator import (
    PointLandmarkCreator,
)
from src.mapping.map_creator.point_map_creator import PointMapCreator
from src.observation.detection.points.orb import ORB
from src.observation.description.points.orb_descriptor import ORBDescriptor
from src.frame.keyframe_selection.every_nth_keyframe_selector import (
    EveryNthKeyframeSelector,
)
from src.metrics import pose_error
from src.observation.mask.multiple_mask import CoordinatesMultipleMask
from src.observation.mask.point.point_clip_mask import PointClipMask
from src.observation.mask.point.point_depth_filter import PointDepthFilter
from src.observation.observation_creator import ObservationsCreator
from src.projection.point_projection import PointProjector
from src.geometry.transform import make_homogeneous_matrix
from src.pose_estimation.rgbd_point_pose_estimator import (
    RGBDPointPoseEstimator,
)
from src.sensor.depth import DepthImage
from src.sensor.rgb import RGBImage
from src.sensor.rgbd import RGBDImage
from src.slam.backend.backend_g2o import G2OPointSLAMBackend
from src.slam.prime_slam import PrimeSLAM
from src.slam.frontend.tracking_frontend import TrackingFrontend


def get_point_cloud(points_3d):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    return point_cloud


def create_point_map(keyframes, intrinsics, color=(1, 0, 0)):
    abs_poses = [kf.world_to_camera_transform for kf in keyframes]
    depths = [kf.sensor_measurement.depth for kf in keyframes]
    h, w = depths[0].depth_map.shape[:2]
    point_coordinates = np.array([[x, y] for y in range(h) for x in range(w)])

    point_coordinates = np.array(point_coordinates)
    projector = PointProjector()

    points_3d_batch = [
        projector.back_project(
            point_coordinates.copy(),
            depth.depth_map,
            depth.depth_scale,
            intrinsics,
            abs_pose,
        )
        for depth, abs_pose in zip(depths, abs_poses)
    ]

    return [
        get_point_cloud(keypoints_3D).paint_uniform_color(color)
        for keypoints_3D in points_3d_batch
    ]


if __name__ == "__main__":
    np.seterr(all="raise")
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        prog="python demo.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--imgs", "-i", metavar="PATH", help="path to images", default="rgb/"
    )

    parser.add_argument(
        "--depths", "-d", metavar="PATH", help="path to depth maps", default="depth/"
    )

    parser.add_argument(
        "--intrinsics",
        "-I",
        metavar="PATH",
        help="path to intrinsics file",
        default="intrinsics.txt",
    )
    parser.add_argument(
        "--poses",
        "-p",
        metavar="PATH",
        help="path to gt poses (for evaluation)",
        default="poses.txt",
    )
    parser.add_argument(
        "--depth-scaler",
        "-D",
        metavar="NUM",
        help="depth map scaler",
        default=5000,
        type=float,
    )
    parser.add_argument(
        "--frames-step",
        "-f",
        metavar="NUM",
        help="step between keyframes",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--save-cloud",
        "-s",
        metavar="BOOL",
        help="save resulting cloud",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        metavar="BOOL",
        help="print metrics",
        default=True,
        type=bool,
    )
    args = parser.parse_args()
    images_path = Path(args.imgs)
    depth_path = Path(args.depths)
    intrinsics_path = Path(args.intrinsics)
    gt_poses_path = Path(args.poses)

    gt_poses = read_poses(gt_poses_path)
    init_pose = gt_poses[0]

    images_paths = sorted(images_path.iterdir())
    depth_paths = sorted(depth_path.iterdir())
    intrinsics = make_homogeneous_matrix(np.genfromtxt(intrinsics_path))
    depth_scaler = args.depth_scaler
    images_number = len(depth_paths)
    step = args.frames_step
    frames_indices = list(range(0, len(images_paths) - 1, step))
    gt_frames_poses = [gt_poses[i] for i in frames_indices]
    point_coordinates_masks = [PointClipMask(), PointDepthFilter()]
    orb_creator = ObservationsCreator(
        ORB(features_number=1000),
        ORBDescriptor(),
        partial(match_descriptors, metric="hamming", max_ratio=0.8),
        PointProjector(),
        RGBDPointPoseEstimator(intrinsics, 30),
        CoordinatesMultipleMask(point_coordinates_masks),
        "orb",
        map_creator=PointMapCreator(),
    )

    images = [io.imread(path) for path in images_paths]
    depths = [io.imread(path) for path in depth_paths]

    frames = [
        RGBDImage(RGBImage(img), DepthImage(depth, intrinsics, depth_scaler))
        for img, depth in zip(images, depths)
    ]
    keyframe_selector = EveryNthKeyframeSelector(n=step)
    observation_creators = [orb_creator]
    slam = PrimeSLAM(
        backend=G2OPointSLAMBackend(intrinsics),
        frontend=TrackingFrontend(observation_creators, keyframe_selector, init_pose),
    )

    for frame in frames:
        slam.process_sensor_data(frame)
    poses = slam.trajectory
    angular_translation_errors = []
    angular_rotation_errors = []
    absolute_translation_errors = []

    for est_pose, gt_pose in zip(poses, gt_frames_poses):
        (
            angular_translation_error_,
            angular_rotation_error_,
            absolute_translation_error_,
        ) = pose_error(gt_pose.transformation, est_pose)
        angular_translation_errors.append(angular_translation_error_)
        angular_rotation_errors.append(angular_rotation_error_)
        absolute_translation_errors.append(absolute_translation_error_)
    if args.verbose:
        print(
            f"Median angular_translation_error: {np.median(angular_translation_errors)}"
        )
        print(f"Median angular_rotation_error: {np.median(angular_rotation_errors)}")
        print(
            f"Median absolute_translation_error: {np.median(absolute_translation_errors)}"
        )

    clouds = create_point_map(slam.frontend.keyframes, intrinsics)
    resulting_cloud = clouds[0]
    for i in range(1, len(clouds)):
        resulting_cloud += clouds[i]
    if args.save_cloud:
        o3d.io.write_point_cloud("resulting_cloud.pcd", resulting_cloud)

    o3d.visualization.draw_geometries([resulting_cloud])
