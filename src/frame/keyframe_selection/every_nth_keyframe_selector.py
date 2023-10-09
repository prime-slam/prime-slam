from src.frame import Frame
from src.frame.keyframe_selection.keyframe_selector import KeyframeSelector

__all__ = ["EveryNthKeyframeSelector"]


class EveryNthKeyframeSelector(KeyframeSelector):
    def __init__(self, n: int):
        self.n = n
        self.counter = 0

    def is_selected(self, keyframe: Frame) -> bool:
        self.counter += 1
        selected = (self.counter // self.n) == 1
        if selected:
            self.counter = 0

        return selected
