from abc import ABC, abstractmethod

from src.frame import Frame

__all__ = ["KeyframeSelector"]


class KeyframeSelector(ABC):
    @abstractmethod
    def is_selected(self, frame: Frame) -> bool:
        pass
