from src.keyframe_selection.keyframe_selector import KeyframeSelector


class EveryNthKeyframeSelector(KeyframeSelector):
    def __init__(self, n):
        self.n = n
        self.counter = 0

    def is_selected(self, keyframe):
        self.counter += 1
        selected = (self.counter // self.n) == 1
        if selected:
            self.counter = 0

        return selected
