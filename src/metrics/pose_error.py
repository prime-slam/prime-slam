import numpy as np
from typing import Tuple

__all__ = [
    "angular_rotation_error",
    "angular_translation_error",
    "absolute_translation_error",
    "pose_error",
]


def angular_rotation_error(first_rotation, second_rotation) -> float:
    cos = (np.trace(first_rotation @ second_rotation.T) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return np.rad2deg(np.abs(np.arccos(cos)))


def angular_translation_error(first_translation, second_translation) -> float:
    n = np.linalg.norm(first_translation) * np.linalg.norm(second_translation)
    return np.rad2deg(
        np.arccos(np.clip(np.dot(first_translation, second_translation) / n, -1.0, 1.0))
    )


def absolute_translation_error(pose_gt, pose_est) -> float:
    delta_pose = pose_gt @ np.linalg.inv(pose_est)
    delta_translation = delta_pose[:3, 3]
    return np.linalg.norm(delta_translation)


def pose_error(pose_gt, pose_est) -> Tuple[float, float, float]:
    rotation_est = pose_est[:3, :3]
    translation_est = pose_est[:3, 3]
    rotation_gt = pose_gt[:3, :3]
    translation_gt = pose_gt[:3, 3]
    angular_translation_error_ = angular_translation_error(
        translation_est, translation_gt
    )
    angular_rotation_error_ = angular_rotation_error(rotation_est, rotation_gt)
    absolute_translation_error_ = absolute_translation_error(pose_gt, pose_est)

    return (
        angular_translation_error_,
        angular_rotation_error_,
        absolute_translation_error_,
    )
