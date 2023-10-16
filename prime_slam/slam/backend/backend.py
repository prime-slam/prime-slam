from abc import ABC, abstractmethod

from prime_slam.slam.graph.factor_graph import FactorGraph


class Backend(ABC):
    @abstractmethod
    def optimize(self, graph: FactorGraph, verbose=False):
        pass
