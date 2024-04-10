from dataclasses import dataclass, field
from typing import List, Set, Tuple

from rtsm.instance import Instance


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

    @classmethod
    def to_json(cls, solutions: Set["Solution"]) -> List[List[str]]:
        return [sol.tests for sol in solutions]
