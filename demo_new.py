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

import sys

sys.path.append("external/voxel_slam")

import numpy as np
import open3d as o3d
from tqdm import tqdm

import argparse
from pathlib import Path

from external.voxel_slam.slam.pipeline import (
    SequentialPipeline,
    SequentialPipelineRuntimeParameters,
)
from octreelib.grid import VisualizationConfig
from prime_slam.data import DataFormatRGBD, DataFormatStereo, DatasetFactory
from prime_slam.geometry import Pose
from prime_slam.observation import (
    FilterChain,
    PointClipFOVFilter,
    PointNonpositiveDepthFilter,
    SuperPoint,
    SuperPointDescriptor,
)
from prime_slam.projection import PointProjector
from prime_slam.slam import (
    EveryNthKeyframeSelector,
    PointMapCreator,
    PrimeSLAM,
    RGBDPointPoseEstimatorG2O,
    SLAMConfig,
    SLAMModuleFactory,
    TrackingFrontend,
)
from prime_slam.slam.tracking.matching.points.superglue import SuperGlue
from prime_slam.slam.tracking.matching.points.superpoint_matcher import (
    SuperPointMatcher,
)
from prime_slam.typing.hints import DetectionMatchingConfig
from prime_slam.utils.configuration_reader import PrimeSLAMConfigurationReader
from prime_slam.utils.utils import write_trajectory


def create_point_cloud(points_3d):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    return point_cloud


def create_superpoint_config():
    super_point_config = DetectionMatchingConfig(
        name="superpoint",
        detector=SuperPoint(),
        descriptor=SuperPointDescriptor(),
        frame_matcher=SuperGlue(),
        map_matcher=SuperPointMatcher(),
    )
    return super_point_config


def create_slam_config(detection_matching_config: DetectionMatchingConfig, intrinsics):
    name = detection_matching_config.name
    projector = PointProjector()
    filters = [PointClipFOVFilter(), PointNonpositiveDepthFilter()]
    pose_estimator = RGBDPointPoseEstimatorG2O(intrinsics)
    map_creator = PointMapCreator(projector, name)
    return SLAMConfig(
        detector=detection_matching_config.detector,
        descriptor=detection_matching_config.descriptor,
        frame_matcher=detection_matching_config.frame_matcher,
        map_matcher=detection_matching_config.map_matcher,
        projector=projector,
        pose_estimator=pose_estimator,
        observations_filter=FilterChain(filters),
        map_creator=map_creator,
        observation_name=name,
    )


def voxel_slam_optimization(
    configuration_reader, pcds_batch, trajectory_batch, output_path, ind
):
    trajectory_batch = [
        np.linalg.inv(pose) @ configuration_reader.cam0_to_lidar
        for pose in trajectory_batch
    ]
    for iteration_ind in range(configuration_reader.patches_iterations):
        voxel_pipeline = SequentialPipeline(
            point_clouds=pcds_batch,
            poses=trajectory_batch,
            subdividers=configuration_reader.subdividers,
            segmenters=configuration_reader.segmenters,
            filters=configuration_reader.filters,
            backend=configuration_reader.backend(
                ind - configuration_reader.patches_step, ind
            ),
        )
        output = voxel_pipeline.run(
            SequentialPipelineRuntimeParameters(
                grid_configuration=configuration_reader.grid_configuration,
                visualization_config=VisualizationConfig(
                    filepath=output_path
                    / f"{ind-configuration_reader.patches_step}-{ind - 1}_{iteration_ind}.html"
                ),
                initial_point_cloud_number=(
                    ind - (ind - configuration_reader.patches_step)
                )
                // 2,
            )
        )
        print(f"Iteration: {iteration_ind}:\n{output}")

        for pose_ind in range(len(trajectory_batch)):
            trajectory_batch[pose_ind] = (
                output.poses[pose_ind] @ trajectory_batch[pose_ind]
            )
    return trajectory_batch


if __name__ == "__main__":
    np.seterr(all="raise")
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        prog="python demo_new.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configuration-path",
        "-c",
        metavar="PATH",
        help=f"path to config",
    )
    args = parser.parse_args()
    available_rgbd_formats = DataFormatRGBD.to_string().split(", ")
    available_stereo_formats = DataFormatStereo.to_string().split(", ")
    available_formats = available_rgbd_formats + available_stereo_formats

    configuration_reader = PrimeSLAMConfigurationReader(args.configuration_path)
    super_point_config = create_superpoint_config()
    data_path = Path(configuration_reader.dataset_path)
    output_path = data_path / "output"
    output_path.mkdir(exist_ok=True)
    data_format = configuration_reader.dataset_type
    if data_format in available_stereo_formats:
        dataset = DatasetFactory.create_from_stereo(
            data_format, data_path, super_point_config, configuration_reader
        )
    elif data_format in available_rgbd_formats:
        dataset = DatasetFactory.create_from_rgbd(data_format, data_path)
    else:
        raise ValueError(
            f"Unsupported data format {data_format}. " f"Expected: {available_formats}"
        )

    keyframe_selector = EveryNthKeyframeSelector(1)
    slam_config = create_slam_config(super_point_config, dataset.intrinsics)
    slam = PrimeSLAM(
        backend=None,
        frontend=TrackingFrontend(
            SLAMModuleFactory([slam_config]),
            keyframe_selector,
            initial_pose=Pose(np.eye(4)),
        ),
    )

    start = configuration_reader.patches_start
    end = configuration_reader.patches_end
    lidar_optimized_poses = np.empty((end - start, 4, 4))

    for i in tqdm(range(start, end)):
        slam.process_sensor_data(dataset[i])
        if (
            (i + 1) % configuration_reader.patches_step == 0 or (i + 1) == end
        ) and data_format == "stereo_lidar":
            trajectory_batch = slam.trajectory[-configuration_reader.patches_step :]
            pcds_batch = dataset.get_point_clouds_batch(
                i + 1 - configuration_reader.patches_step, i + 1
            )
            lidar_optimized_poses[
                i + 1 - configuration_reader.patches_step : i + 1
            ] = voxel_slam_optimization(
                configuration_reader, pcds_batch, trajectory_batch, output_path, i + 1
            )

    write_trajectory(
        output_path / "lidar_optimized_trajectory.txt", lidar_optimized_poses
    )
    write_trajectory(output_path / "cam0_unoptimized_trajectory.txt", slam.trajectory)
    o3d.io.write_point_cloud(
        str(output_path / "feat_map.pcd"),
        create_point_cloud(
            slam.frontend.map.get_positions(slam_config.observation_name)
        ),
    )
