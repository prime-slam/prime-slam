from abc import ABC, abstractmethod

from src.graph.factor_graph import FactorGraph


class Backend(ABC):
    @abstractmethod
    def optimize(self, graph: FactorGraph, verbose=False):
        pass
