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
from scipy.spatial.transform import Rotation as R

from abc import ABC
from pathlib import Path
from typing import List

from prime_slam.data.rgbd_dataset import RGBDDataset
from prime_slam.geometry import Pose, make_euclidean_transform
from prime_slam.typing.hints import Transformation


class TUMRGBDDatasetBase(RGBDDataset, ABC):
    def __init__(
        self,
        data_path: Path,
        rgb_directory: Path = Path("rgb"),
        depth_directory: Path = Path("depth"),
        gt_poses_file: Path = Path("groundtruth.txt"),
        intrinsics: Transformation = np.array(
            [
                [525.0, 0, 319.5, 0],
                [0, 525.0, 239.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        depth_factor: float = 5000,
    ):
        self.rgb_base_path = data_path / rgb_directory
        self.rgb_paths = sorted(self.rgb_base_path.iterdir())
        self.depth_base_path = data_path / depth_directory
        self.depth_paths = sorted(self.depth_base_path.iterdir())
        self.gt_poses_path = data_path / gt_poses_file
        self._gt_poses = self.read_gt_poses(self.gt_poses_path)
        self._intrinsics = intrinsics
        self.depth_factor = depth_factor

    @property
    def gt_poses(self) -> List[Pose]:
        return self._gt_poses

    @property
    def intrinsics(self) -> Transformation:
        return self._intrinsics

    def __len__(self) -> int:
        return len(self.rgb_paths)

    @staticmethod
    def read_gt_poses(gt_poses_path: Path, comment_symbol="#") -> List[Pose]:
        gt_poses = []
        for line in gt_poses_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(comment_symbol):
                continue
            _, tx, ty, tz, qx, qy, qz, qw = line.split(" ")
            translation = np.array([tx, ty, tz], dtype=float)
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            gt_poses.append(
                Pose(make_euclidean_transform(rotation, translation)).inverse()
            )
        return gt_poses
