import numpy as np

from src.typing.hints import Array3x3, Array4x4


def make_euclidean_transform(rotation_matrix, translation):
    transform = np.hstack([rotation_matrix, translation.reshape(-1, 1)])
    transform = np.vstack([transform, [0, 0, 0, 1]])
    return transform


def make_homogeneous_matrix(matrix: Array3x3[float]) -> Array4x4[float]:
    matrix_homo = np.hstack([matrix, np.zeros((3, 1))])
    matrix_homo = np.vstack([matrix_homo, [0, 0, 0, 1]])
    return matrix_homo
