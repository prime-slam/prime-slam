from abc import ABC, abstractmethod


class Keyobject(ABC):
    @property
    @abstractmethod
    def coordinates(self):
        pass

    @property
    @abstractmethod
    def uncertainty(self):
        pass

    @property
    @abstractmethod
    def data(self):
        pass
