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
from skimage import io

from pathlib import Path

from prime_slam.data.constants import TUM_DEFAULT_INTRINSICS, TUM_DEPTH_FACTOR
from prime_slam.data.tum_rgbd_dataset_base import TUMRGBDDatasetBase
from prime_slam.sensor import DepthImage, RGBDImage, RGBImage
from prime_slam.typing.hints import ArrayNx2, Transformation

__all__ = ["TUMRGBDDataset"]


class TUMRGBDDataset(TUMRGBDDatasetBase):
    def __init__(
        self,
        data_path: Path,
        rgb_directory: Path = Path("rgb"),
        depth_directory: Path = Path("depth"),
        gt_poses_file: Path = Path("groundtruth.txt"),
        intrinsics: Transformation = TUM_DEFAULT_INTRINSICS,
        depth_factor: float = TUM_DEPTH_FACTOR,
    ):
        super().__init__(
            data_path,
            rgb_directory,
            depth_directory,
            gt_poses_file,
            intrinsics,
            depth_factor,
        )
        self.rgb_depth_associations = self.__create_rgb_depth_association()

    def __getitem__(self, index) -> RGBDImage:
        rgb_index, depth_index = self.rgb_depth_associations[index]
        rgb_image = RGBImage(io.imread(self.rgb_paths[rgb_index]))
        depth_image = DepthImage(
            io.imread(self.depth_paths[depth_index]) / self.depth_factor,
            self.intrinsics,
        )
        return RGBDImage(rgb_image, depth_image)

    def __create_rgb_depth_association(self) -> ArrayNx2[int]:
        timestamps_rgb = np.array(
            list(map(lambda path: float(path.stem), self.rgb_paths))
        )
        timestamps_depth = np.array(
            list(map(lambda path: float(path.stem), self.depth_paths))
        )
        depth_dist = np.abs(timestamps_rgb[:, None] - timestamps_depth[None])
        closest_depths = np.argmin(depth_dist, 1)

        return np.column_stack([np.arange(len(self.rgb_paths)), closest_depths])
