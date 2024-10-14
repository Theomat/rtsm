from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from rtsm.instance import Instance

import numpy as np


@dataclass(frozen=True)
class Solution:
    """
    Represents a solution to an instance of the RTSM problem.
    """

    instance: Instance = field(hash=False)
    tests: Tuple[str]

    def cost(self) -> int:
        """
        Return the cost of this solution.
        """
        return len(self.tests)

    def measure_performances(self) -> Dict[str, Tuple[float, float]]:
        mask = [t in self.tests for t in self.instance.tests]
        remaining = np.sum(self.instance.performance_matrix[:, :, mask], axis=(1, -1))
        total = np.sum(self.instance.performance_matrix, axis=(1, -1))
        return {
            perf: (remaining[i], total[i])
            for i, perf in enumerate(self.instance.performances)
        }

    @classmethod
    def to_json(cls, solutions: Set["Solution"]) -> List[List[str]]:
        return [sorted(sol.tests) for sol in solutions]
