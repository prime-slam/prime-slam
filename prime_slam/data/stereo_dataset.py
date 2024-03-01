# Copyright (c) 2023, Moskalenko Ivan, Kirill Ivanov, Anastasiia Kornilova
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

import cv2
import numpy as np

from pathlib import Path

from prime_slam.data.rgbd_dataset import RGBDDataset
from prime_slam.observation import ObservationData
from prime_slam.observation.description.descriptor import Descriptor
from prime_slam.observation.detection.detector import Detector
from prime_slam.observation.observation import Observation
from prime_slam.sensor import DepthImage, RGBDImage, RGBImage
from prime_slam.slam.tracking.matching.frame_matcher import ObservationsMatcher
from prime_slam.typing.hints import DetectionMatchingConfig, Transformation
from prime_slam.utils.configuration_reader import PrimeSLAMConfigurationReader

__all__ = ["StereoDataset"]


class StereoDataset(RGBDDataset):
    def __init__(
        self,
        data_path: Path,
        detection_matching_config: DetectionMatchingConfig,
        configuration_reader: PrimeSLAMConfigurationReader,
    ):
        self.cam0_base_path = data_path / "cam0"
        self.cam0_paths = sorted(self.cam0_base_path.iterdir())
        self.cam1_base_path = data_path / "cam1"
        self.cam1_paths = sorted(self.cam1_base_path.iterdir())

        K1 = configuration_reader.cam0_intrinsics[:3, :3]
        K2 = configuration_reader.cam1_intrinsics[:3, :3]
        D1 = configuration_reader.cam0_distortion
        D2 = configuration_reader.cam1_distortion
        self._img_size = cv2.imread(str(self.cam0_paths[0])).shape[1::-1]
        R = configuration_reader.cam0_to_cam1[:3, :3]
        T = configuration_reader.cam0_to_cam1[:3, 3]
        self._baseline = np.linalg.norm(T)

        rectify_flags = cv2.CALIB_ZERO_DISPARITY
        R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2, self._img_size, R, T, flags=rectify_flags
        )
        self._intrinsics = P1
        self._map1x, self._map1y = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R1, P1, self._img_size, cv2.CV_32FC1
        )
        self._map2x, self._map2y = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, R2, P2, self._img_size, cv2.CV_32FC1
        )

        self._detector: Detector = detection_matching_config.detector
        self._matcher: ObservationsMatcher = detection_matching_config.frame_matcher
        self._descriptor: Descriptor = detection_matching_config.descriptor
        self._configuration_reader = configuration_reader

    def __getitem__(self, index) -> RGBDImage:
        cam0_img = cv2.imread(str(self.cam0_paths[index]))
        cam1_img = cv2.imread(str(self.cam1_paths[index]))
        cam0_img = cv2.remap(cam0_img, self._map1x, self._map1y, cv2.INTER_LINEAR)
        cam1_img = cv2.remap(cam1_img, self._map2x, self._map2y, cv2.INTER_LINEAR)

        cam0_img = RGBImage(cam0_img)
        cam1_img = RGBImage(cam1_img)
        cam0_img = RGBDImage(cam0_img, None)
        cam1_img = RGBDImage(cam1_img, None)

        cam0_kpts = self._detector.detect(cam0_img)
        cam1_kpts = self._detector.detect(cam1_img)
        cam0_descs = self._descriptor.descript(cam0_kpts, cam0_img)
        cam1_descs = self._descriptor.descript(cam1_kpts, cam1_img)
        cam0_observations = []
        for kpt, desc in zip(cam0_kpts, cam0_descs):
            cam0_observations.append(Observation(kpt, desc))
        cam1_observations = []
        for kpt, desc in zip(cam1_kpts, cam1_descs):
            cam1_observations.append(Observation(kpt, desc))
        cam0_observation_data = ObservationData(cam0_observations, "cam0", cam0_img)
        cam1_observation_data = ObservationData(cam1_observations, "cam1", cam1_img)

        matches = self._matcher.match_observations(
            cam0_observation_data, cam1_observation_data
        )

        f = self._intrinsics[0, 0]
        depth_img = np.zeros(self._img_size[::-1], dtype=np.float32)
        for match in matches:
            cam1_index, cam0_index = match
            disparity = np.abs(
                cam0_kpts[cam0_index].coordinates[0]
                - cam1_kpts[cam1_index].coordinates[0]
            )
            if disparity == 0:
                continue
            depth = (f * self._baseline) / disparity
            x, y = cam0_kpts[cam0_index].coordinates
            depth_img[round(y), round(x)] = depth

        cam0_img.depth = DepthImage(depth_img, self._intrinsics)

        return cam0_img

    def __len__(self) -> int:
        return len(self.cam0_paths)

    @property
    def gt_poses(self) -> None:
        return None

    @property
    def intrinsics(self) -> Transformation:
        return self._intrinsics
