import numpy as np
import numpy.typing as npt

from typing import Annotated, Literal, TypeVar

__all__ = [
    "ArrayNxM",
    "ArrayN",
    "ArrayNx2",
    "ArrayNx3",
    "ArrayNx4",
    "ArrayNx2x2",
    "ArrayNxMx2",
    "ArrayNxN",
    "Array4x4",
    "Array3",
    "Array3x3",
    "ArrayNx6",
    "Rotation",
    "Translation",
    "Transformation",
]

DType = TypeVar("DType", bound=np.generic)

ArrayNxM = Annotated[npt.NDArray[DType], Literal["N", "M"]]

ArrayNxN = Annotated[npt.NDArray[DType], Literal["N", "N"]]

ArrayNxMx2 = Annotated[npt.NDArray[DType], Literal["N", "M", 2]]

ArrayN = Annotated[npt.NDArray[DType], Literal["N"]]

Array3 = Annotated[npt.NDArray[DType], Literal[3]]

ArrayNx2 = Annotated[npt.NDArray[DType], Literal["N", 2]]

ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", 3]]

ArrayNx4 = Annotated[npt.NDArray[DType], Literal["N", 4]]

ArrayNx6 = Annotated[npt.NDArray[DType], Literal["N", 6]]

Array4x4 = Annotated[npt.NDArray[DType], Literal[4, 4]]

Array3x3 = Annotated[npt.NDArray[DType], Literal[4, 4]]

ArrayNx2x2 = Annotated[npt.NDArray[DType], Literal["N", 2, 2]]

Transformation = Array4x4[float]

Rotation = Array3x3[float]

Translation = Array3[float]
