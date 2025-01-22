from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from rtsm.instance import Instance

import numpy as np


@dataclass
class Solution:
    """
    Represents a solution to an instance of the RTSM problem.
    """

    instance: Instance = field(hash=False)
    tests: Tuple[str]
    __mask: np.ndarray = field(
        default_factory=lambda: np.ones((1,)), compare=False, repr=False, hash=False
    )

    def __hash__(self) -> int:
        return hash(self.tests)

    def cost(self) -> float:
        """
        Return the cost of this solution.
        """
        return np.sum(self.instance.costs[self.to_mask()])

    def measure_performances(self) -> Dict[str, Tuple[float, float]]:
        mask = self.to_mask()
        remaining = np.sum(self.instance.performance_matrix[:, :, mask], axis=(1, -1))
        total = np.sum(self.instance.performance_matrix, axis=(1, -1))
        return {
            perf: (remaining[i], total[i])
            for i, perf in enumerate(self.instance.performances)
        }

    def to_mask(self) -> np.ndarray:
        if len(self.__mask) != len(self.instance.tests):
            self.__mask = np.array([t in self.tests for t in self.instance.tests])
        return self.__mask

    @classmethod
    def to_json(cls, solutions: Set["Solution"]) -> List[List[str]]:
        return [sorted(sol.tests) for sol in solutions]
