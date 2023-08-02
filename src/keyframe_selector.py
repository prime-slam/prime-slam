class EveryNthKeyframeSelector:
    def __init__(self, n):
        self.n = n
        self.counter = 0

    def select(self, keyframe):
        self.counter += 1
        selected = (self.counter // self.n) == 1
        if selected:
            self.counter = 0

        return selected
