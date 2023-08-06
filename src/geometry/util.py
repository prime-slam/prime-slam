import numpy as np


def clip_lines(lines, height: float, width: float):
    x_index = [0, 2]
    lines[..., x_index] = np.clip(lines[..., x_index], 0, width - 1)
    y_index = [1, 3]
    lines[..., y_index] = np.clip(lines[..., y_index], 0, height - 1)
    return lines
