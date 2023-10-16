import numpy as np

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import List

from prime_slam.geometry.pose import Pose
from prime_slam.geometry.transform import (
    make_euclidean_transform,
)

__all__ = ["read_poses"]


def read_poses(poses_path: Path, comment_symbol: str = "#") -> List[Pose]:
    euclidean_transforms = []
    for line in poses_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(comment_symbol):
            continue
        _, tx, ty, tz, qx, qy, qz, qw = line.split(" ")
        translation = np.array([tx, ty, tz], dtype=float)
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        euclidean_transforms.append(
            Pose(make_euclidean_transform(rotation, translation)).inverse()
        )
    return euclidean_transforms
