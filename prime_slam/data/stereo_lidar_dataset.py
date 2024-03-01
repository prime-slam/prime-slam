# Copyright (c) 2024, Moskalenko Ivan, Kirill Ivanov, Anastasiia Kornilova
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

from pathlib import Path

from prime_slam.data.stereo_dataset import StereoDataset
from prime_slam.typing.hints import DetectionMatchingConfig
from prime_slam.utils.configuration_reader import PrimeSLAMConfigurationReader

__all__ = ["StereoLidarDataset"]


class StereoLidarDataset(StereoDataset):
    def __init__(
        self,
        data_path: Path,
        detection_matching_config: DetectionMatchingConfig,
        configuration_reader: PrimeSLAMConfigurationReader,
    ):
        super().__init__(data_path, detection_matching_config, configuration_reader)
        self.point_clouds_base_path = data_path / "pcds"
        self.point_clouds_paths = sorted(self.point_clouds_base_path.iterdir())

        images_timestamps = np.array([float(path.stem) for path in self.cam0_paths])
        clouds_timestamps = np.array(
            [float(path.stem) for path in self.point_clouds_paths]
        )

        if len(clouds_timestamps) != len(images_timestamps):
            new_cam0_paths = []
            new_cam1_paths = []
            for cloud_ts in clouds_timestamps:
                nearest_image = np.argmin(np.abs(images_timestamps - cloud_ts))
                new_cam0_paths.append(self.cam0_paths[nearest_image])
                new_cam1_paths.append(self.cam1_paths[nearest_image])
            self.cam0_paths = new_cam0_paths
            self.cam1_paths = new_cam1_paths

    def get_point_clouds_batch(self, start, end) -> list[o3d.geometry.PointCloud]:
        return [
            o3d.io.read_point_cloud(str(pcd))
            for pcd in self.point_clouds_paths[start:end]
        ]
