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

from pathlib import Path
from skimage import io
from typing import List, Dict

from prime_slam.data.constants import ICL_NUIM_DEFAULT_INTRINSICS, TUM_DEPTH_FACTOR
from prime_slam.data.rgbd_dataset import RGBDDataset
from prime_slam.data.tum_rgbd_dataset_base import TUMRGBDRGBDDatasetBase
from prime_slam.geometry import Pose, normalize
from prime_slam.sensor import RGBDImage, RGBImage, DepthImage
from prime_slam.typing.hints import Transformation, ArrayN

__all__ = ["ICLNUIMRGBDDataset"]


class ICLNUIMRGBDDataset(RGBDDataset):
    def __init__(
        self,
        data_path: Path,
        rgb_extension: str = ".png",
        depth_extension: str = ".depth",
        camera_parameters_extension: str = ".txt",
        intrinsics: Transformation = ICL_NUIM_DEFAULT_INTRINSICS,
    ):
        self.rgb_paths = sorted(data_path.rglob(f"*{rgb_extension}"))
        self.depth_paths = sorted(data_path.rglob(f"*{depth_extension}"))
        self.camera_param_paths = sorted(
            data_path.rglob(f"*{camera_parameters_extension}")
        )
        self._gt_poses = self.__create_gt_poses()
        self._intrinsics = intrinsics

    def __getitem__(self, index) -> RGBDImage:
        rgb_image = RGBImage(io.imread(self.rgb_paths[index]))
        height, width = rgb_image.image.shape[:2]
        depth_image = DepthImage(
            self.__convert_euclidean_dists_to_depth_map(
                np.genfromtxt(self.depth_paths[index]), height, width
            ),
            self.intrinsics,
        )
        return RGBDImage(rgb_image, depth_image)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    @property
    def gt_poses(self) -> List[Pose]:
        return self._gt_poses

    @property
    def intrinsics(self) -> Transformation:
        return self._intrinsics

    @staticmethod
    def create_tum_format(
        data_path: Path,
        rgb_directory: Path = Path("rgb"),
        depth_directory: Path = Path("depth"),
        gt_poses_file: Path = Path("groundtruth.txt"),
        intrinsics: Transformation = ICL_NUIM_DEFAULT_INTRINSICS,
        depth_factor: float = TUM_DEPTH_FACTOR,
    ) -> "ICLNUIMTUMFormatDataset":
        return ICLNUIMTUMFormatDataset(
            data_path,
            rgb_directory,
            depth_directory,
            gt_poses_file,
            intrinsics,
            depth_factor,
        )

    def __create_gt_poses(self):
        gt_poses = [
            self.__create_pose(self.__read_camera_file(camera_file))
            for camera_file in self.camera_param_paths
        ]

        return gt_poses

    def __convert_euclidean_dists_to_depth_map(
        self, euclidean_dists: ArrayN[float], height: int, width: int
    ):
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        shape = (height, width)

        euclidean_dists = euclidean_dists.reshape(shape)
        depth_map = np.zeros(shape, dtype=float)

        for y in range(height):
            for x in range(width):
                x_dir = (x - cx) / fx
                y_dir = (y - cy) / fy
                depth_map[y, x] = euclidean_dists[y, x] / np.sqrt(
                    x_dir * x_dir + y_dir * y_dir + 1
                )
        return depth_map

    @staticmethod
    def __create_pose(camera_parameters) -> Pose:
        camera_direction = camera_parameters["cam_dir"]
        camera_position = camera_parameters["cam_pos"]
        camera_up = camera_parameters["cam_up"]
        z = normalize(camera_direction)
        x = normalize(np.cross(camera_up, z))
        y = np.cross(z, x)

        rotation = np.column_stack([x, y, z])
        translation = camera_position

        return Pose.from_rotation_and_translation(rotation, translation)

    @staticmethod
    def __read_camera_file(file: Path) -> Dict:
        params = {}
        for line in file.read_text().splitlines():
            name, value = map(str.strip, line.split("="))
            if value.startswith("["):
                value = np.array(eval(value[:-2]))
            else:
                value = eval(value[:-1])
            params[name] = value
        return params


class ICLNUIMTUMFormatDataset(TUMRGBDRGBDDatasetBase):
    def __init__(
        self,
        data_path: Path,
        rgb_directory: Path = Path("rgb"),
        depth_directory: Path = Path("depth"),
        gt_poses_file: Path = Path("groundtruth.txt"),
        intrinsics: Transformation = ICL_NUIM_DEFAULT_INTRINSICS,
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

    def __getitem__(self, index) -> RGBDImage:
        rgb_image = RGBImage(io.imread(self.rgb_paths[index]))
        depth_image = DepthImage(
            io.imread(self.depth_paths[index]) / self.depth_factor,
            self.intrinsics,
        )
        return RGBDImage(rgb_image, depth_image)
