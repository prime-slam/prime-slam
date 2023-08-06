from abc import ABC, abstractmethod


class KeyframeSelector(ABC):
    @abstractmethod
    def is_selected(self, keyframe):
        pass
