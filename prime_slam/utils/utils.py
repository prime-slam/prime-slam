import mrob

from pathlib import Path


def write_trajectory(path_to_save: Path, trajectory):
    with open(path_to_save, "w") as f:
        for pose in trajectory:
            qx, qy, qz, qw = mrob.geometry.so3_to_quat(pose[:3, :3])
            x, y, z = pose[:3, 3]
            f.write(f"{x} {y} {z} {qx} {qy} {qz} {qw}\n")
