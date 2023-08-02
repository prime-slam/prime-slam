import numpy as np

from pathlib import Path
from scipy.spatial.transform import Rotation as R

from src.geometry.transform import (
    make_euclidean_transform,
)


def read_poses(poses_path: Path, comment_symbol: str = "#"):
    euclidean_transforms = []
    for line in poses_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(comment_symbol):
            continue
        _, tx, ty, tz, qx, qy, qz, qw = line.split(" ")
        translation = np.array([tx, ty, tz], dtype=float)
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        euclidean_transforms.append(
            np.linalg.inv(make_euclidean_transform(rotation, translation))
        )
    return euclidean_transforms
