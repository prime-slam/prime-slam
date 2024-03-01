import copy
import numpy as np
import mrob

from pathlib import Path

from external.voxel_slam.slam.pipeline import YAMLConfigurationReader


class PrimeSLAMConfigurationReader(YAMLConfigurationReader):
    @property
    def dataset_type(self) -> str:
        """
        Represents the type of the dataset

        :return: Dataset type
        """
        try:
            dataset = copy.deepcopy(self._configuration["dataset"])
            dataset_type = dataset["type"]
        except KeyError as e:
            raise ValueError(f"{e} must be set")

        return dataset_type

    @property
    def cam0_intrinsics(self) -> np.ndarray:
        """
        Represents the intrinsics of the cam0

        :return: 4x4 intrinsics matrix
        """
        try:
            intrinsics = copy.deepcopy(self._configuration["intrinsics"])
            cam0_intrinsics = intrinsics["cam0"]
        except KeyError as e:
            raise ValueError(f"{e} must be set")

        return np.array(cam0_intrinsics)

    @property
    def cam1_intrinsics(self) -> np.ndarray:
        """
        Represents the intrinsics of the cam1

        :return: 4x4 intrinsics matrix
        """
        try:
            intrinsics = copy.deepcopy(self._configuration["intrinsics"])
            cam1_intrinsics = intrinsics["cam1"]
        except KeyError as e:
            raise ValueError(f"{e} must be set")

        return np.array(cam1_intrinsics)

    @property
    def cam0_distortion(self) -> np.ndarray:
        """
        Represents the distortion of the cam0

        :return: [k1, k2, k3, k4] fisheye distortion coefficients
        """
        try:
            distortion = copy.deepcopy(self._configuration["distortion"])
            cam0_distortion = distortion["cam0"]
        except KeyError as e:
            raise ValueError(f"{e} must be set")

        return np.array(cam0_distortion)

    @property
    def cam1_distortion(self) -> np.ndarray:
        """
        Represents the distortion of the cam1

        :return: [k1, k2, k3, k4] fisheye distortion coefficients
        """
        try:
            distortion = copy.deepcopy(self._configuration["distortion"])
            cam1_distortion = distortion["cam1"]
        except KeyError as e:
            raise ValueError(f"{e} must be set")

        return np.array(cam1_distortion)

    @property
    def cam0_to_cam1(self) -> np.ndarray:
        """
        Represents the transformation between cam0 and cam1

        :return: 4x4 transformation matrix from cam0 to cam1
        """
        try:
            cam0_to_cam1 = copy.deepcopy(self._configuration["cam0_to_cam1"])
            quat = cam0_to_cam1["quat"]
            translation = cam0_to_cam1["translation"]
            pose = np.eye(4)
            pose[:3, :3] = mrob.geometry.quat_to_so3(quat)
            pose[:3, 3] = translation
        except KeyError as e:
            raise ValueError(f"{e} must be set")

        return pose

    @property
    def cam0_to_lidar(self) -> np.ndarray:
        """
        Represents the transformation between cam0 and LiDAR

        :return: 4x4 transformation matrix from cam0 to LiDAR
        """
        try:
            cam0_to_lidar = copy.deepcopy(self._configuration["cam0_to_lidar"])
            quat = cam0_to_lidar["quat"]
            translation = cam0_to_lidar["translation"]
            pose = np.eye(4)
            pose[:3, :3] = mrob.geometry.quat_to_so3(quat)
            pose[:3, 3] = translation
        except KeyError as e:
            raise ValueError(f"{e} must be set")

        return pose
