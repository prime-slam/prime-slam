import numpy as np

from prime_slam.typing.hints import ArrayNx4, ArrayN


__all__ = ["clip_lines", "normalize"]


def clip_lines(lines: ArrayNx4[float], height: float, width: float) -> ArrayNx4[float]:
    x_index = [0, 2]
    lines[..., x_index] = np.clip(lines[..., x_index], 0, width - 1)
    y_index = [1, 3]
    lines[..., y_index] = np.clip(lines[..., y_index], 0, height - 1)
    return lines


def normalize(vector: ArrayN[float], epsilon: float = 1.0e-10) -> ArrayN[float]:
    norm = np.linalg.norm(vector)
    return vector if norm >= epsilon else vector