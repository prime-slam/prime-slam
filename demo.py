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

import argparse
import numpy as np
import open3d as o3d

from functools import partial
from pathlib import Path
from skimage.feature import match_descriptors

from prime_slam.data import DatasetFactory, DataFormat
from prime_slam.metrics.pose_error import pose_error
from prime_slam.observation import (
    ORB,
    ORBDescriptor,
    FilterChain,
    PointClipFOVFilter,
    PointNonpositiveDepthFilter,
)
from prime_slam.projection import PointProjector
from prime_slam.slam import (
    SLAMConfig,
    RGBDPointPoseEstimator,
    PointMapCreator,
    EveryNthKeyframeSelector,
    PrimeSLAM,
    G2OPointSLAMBackend,
    TrackingFrontend,
    SLAMModuleFactory,
)


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
            intrinsics,
            abs_pose,
        )
        for depth, abs_pose in zip(depths, abs_poses)
    ]

    return [
        get_point_cloud(keypoints_3D).paint_uniform_color(color)
        for keypoints_3D in points_3d_batch
    ]


def create_orb_config(intrinsics):
    name = "orb"
    projector = PointProjector()
    return SLAMConfig(
        detector=ORB(features_number=1000),
        descriptor=ORBDescriptor(),
        matcher=partial(match_descriptors, metric="hamming", max_ratio=0.8),
        projector=projector,
        pose_estimator=RGBDPointPoseEstimator(intrinsics, 30),
        observations_filter=FilterChain(point_filters),
        map_creator=PointMapCreator(projector, name),
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
    step = args.frames_step
    frames_indices = list(range(0, images_number - 1, step))
    gt_frames_poses = [gt_poses[i] for i in frames_indices]

    point_filters = [PointClipFOVFilter(), PointNonpositiveDepthFilter()]
    orb_config = create_orb_config(dataset.intrinsics)
    keyframe_selector = EveryNthKeyframeSelector(n=step)
    slam_configs = [orb_config]
    slam = PrimeSLAM(
        backend=G2OPointSLAMBackend(dataset.intrinsics),
        frontend=TrackingFrontend(
            SLAMModuleFactory(slam_configs), keyframe_selector, init_pose
        ),
    )
    for data in dataset:
        slam.process_sensor_data(data)

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

    clouds = create_point_map(slam.frontend.keyframes, dataset.intrinsics)
    resulting_cloud = clouds[0]
    for i in range(1, len(clouds)):
        resulting_cloud += clouds[i]
    if args.save_cloud:
        o3d.io.write_point_cloud(args.cloud_save_path, resulting_cloud)

    o3d.visualization.draw_geometries([resulting_cloud])
