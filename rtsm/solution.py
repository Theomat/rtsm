from dataclasses import dataclass
from typing import List, Set

from rtsm.instance import Instance


@dataclass
class Solution:
    """
    Represents a solution to an instance of the RTSM problem.
    """

    instance: Instance
    tests: List[str]

    def cost(self) -> int:
        """
        Return the cost of this solution.
        """
        return len(self.tests)

    @classmethod
    def to_json(cls, solutions: Set["Solution"]) -> List[List[str]]:
        return [sol.tests for sol in solutions]
