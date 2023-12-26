# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import open3d as o3d
from skimage.feature import match_descriptors
from tqdm import tqdm

import argparse
from functools import partial
from pathlib import Path

from prime_slam.data import DataFormat, DatasetFactory
from prime_slam.metrics.pose_error import pose_error
from prime_slam.observation import (
    LBD,
    LSD,
    ORB,
    SIFT,
    FilterChain,
    ORBDescriptor,
    PointClipFOVFilter,
    PointNonpositiveDepthFilter,
    SIFTDescriptor,
    SuperPoint,
    SuperPointDescriptor,
)
from prime_slam.projection import LineProjector, PointProjector
from prime_slam.slam import (
    EveryNthKeyframeSelector,
    G2OPointBackend,
    LineMapCreator,
    MROBPointBackend,
    PointMapCreator,
    PrimeSLAM,
    RGBDPointPoseEstimatorG2O,
    SLAMConfig,
    SLAMModuleFactory,
    TrackingFrontend,
)
from prime_slam.slam.backend.line_backend_mrob import MROBLineBackend
from prime_slam.slam.tracking.matching.default_observations_matcher import (
    DefaultMatcher,
)
from prime_slam.slam.tracking.matching.points.superglue import SuperGlue
from prime_slam.slam.tracking.matching.points.superpoint_matcher import (
    SuperPointMatcher,
)
from prime_slam.slam.tracking.pose_estimation.rgbd_line_pose_estimator_mrob import (
    RGBDLinePoseEstimatorMROB,
)
from prime_slam.slam.tracking.pose_estimation.rgbd_point_pose_estimator_mrob import (
    RGBDPointPoseEstimatorMROB,
)


def create_point_cloud(points_3d):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    return point_cloud


def create_line_set(lines_3d):
    line_set = o3d.geometry.LineSet()
    points = []
    lines = []
    lines_3d = lines_3d.reshape(-1, 2, 3)
    for i, edges in enumerate(lines_3d):
        points.append(edges[0])
        points.append(edges[1])
        lines.append((2 * i, 2 * i + 1))

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set


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
            intrinsics,
            abs_pose,
        )
        for depth, abs_pose in zip(depths, abs_poses)
    ]

    return [
        create_point_cloud(keypoints_3D).paint_uniform_color(color)
        for keypoints_3D in points_3d_batch
    ]


def create_orb_config(intrinsics):
    name = "orb"
    projector = PointProjector()
    point_filters = [PointClipFOVFilter(), PointNonpositiveDepthFilter()]
    matcher = DefaultMatcher(partial(match_descriptors, metric="hamming", max_ratio=1))
    return SLAMConfig(
        detector=ORB(features_number=1000),
        descriptor=ORBDescriptor(),
        frame_matcher=matcher,
        map_matcher=matcher,
        projector=projector,
        pose_estimator=RGBDPointPoseEstimatorMROB(intrinsics),
        observations_filter=FilterChain(point_filters),
        map_creator=PointMapCreator(projector, name),
        observation_name=name,
    )


def create_sift_config(intrinsics):
    name = "sift"
    projector = PointProjector()
    point_filters = [PointClipFOVFilter(), PointNonpositiveDepthFilter()]
    matcher = DefaultMatcher(partial(match_descriptors, max_ratio=1))
    return SLAMConfig(
        detector=SIFT(1000),
        descriptor=SIFTDescriptor(),
        frame_matcher=matcher,
        map_matcher=matcher,
        projector=projector,
        pose_estimator=RGBDPointPoseEstimatorMROB(intrinsics),
        observations_filter=FilterChain(point_filters),
        map_creator=PointMapCreator(projector, name),
        observation_name=name,
    )


def create_superpoint_config(intrinsics, use_super_glue=True):
    name = "superpoint"
    projector = PointProjector()
    point_filters = [PointClipFOVFilter(), PointNonpositiveDepthFilter()]
    superpoint_matcher = SuperPointMatcher()
    return SLAMConfig(
        detector=SuperPoint(),
        descriptor=SuperPointDescriptor(),
        frame_matcher=SuperGlue() if use_super_glue else superpoint_matcher,
        map_matcher=superpoint_matcher,
        projector=projector,
        pose_estimator=RGBDPointPoseEstimatorG2O(intrinsics),
        observations_filter=FilterChain(point_filters),
        map_creator=PointMapCreator(projector, name),
        observation_name=name,
    )


def create_lsd_config(intrinsics):
    name = "lsd"
    projector = LineProjector()
    matcher = DefaultMatcher(partial(match_descriptors, max_ratio=1))
    return SLAMConfig(
        detector=LSD(),
        descriptor=LBD(),
        frame_matcher=matcher,
        map_matcher=matcher,
        projector=projector,
        pose_estimator=RGBDLinePoseEstimatorMROB(intrinsics),
        observations_filter=FilterChain([]),
        map_creator=LineMapCreator(projector, name),
        observation_name=name,
    )


if __name__ == "__main__":
    np.seterr(all="raise")
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        prog="python demo.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", "-d", metavar="PATH", help="path to data", default="data/"
    )
    parser.add_argument(
        "--data-format",
        "-D",
        metavar="STR",
        help=f"data format: {DataFormat.to_string()}",
        default=DataFormat.icl_tum.name,
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
        "--cloud-save-path",
        "-S",
        metavar="PATH",
        help="path to the saved cloud",
        default="resulting_cloud.pcd",
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
    data_format = args.data_format
    data_path = Path(args.data)
    dataset = DatasetFactory.create(data_format, data_path)

    gt_poses = dataset.gt_poses
    init_pose = gt_poses[0]
    images_number = len(dataset)

    orb_config = create_orb_config(dataset.intrinsics)
    sift_config = create_sift_config(dataset.intrinsics)
    super_point_config = create_superpoint_config(dataset.intrinsics)
    lsd_config = create_lsd_config(dataset.intrinsics)
    keyframe_selector = EveryNthKeyframeSelector(10)
    slam_config = super_point_config
    slam = PrimeSLAM(
        backend=G2OPointBackend(dataset.intrinsics),
        frontend=TrackingFrontend(
            SLAMModuleFactory([slam_config]), keyframe_selector, init_pose
        ),
    )
    for data in tqdm(dataset):
        slam.process_sensor_data(data)

    poses = slam.trajectory
    angular_translation_errors = []
    angular_rotation_errors = []
    absolute_translation_errors = []
    keyframe_identifiers = [kf.identifier for kf in slam.frontend.keyframes][:-1]
    gt_frames_poses = [gt_poses[i] for i in keyframe_identifiers]
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

    clouds = create_point_map(slam.frontend.keyframes, dataset.intrinsics)
    resulting_cloud = clouds[0]
    for i in range(1, len(clouds)):
        resulting_cloud += clouds[i]
    if args.save_cloud:
        o3d.io.write_point_cloud(args.cloud_save_path, resulting_cloud)
    o3d.visualization.draw_geometries(
        [
            create_point_cloud(
                slam.frontend.map.get_positions(slam_config.observation_name)
            )
        ]
    )
