from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np


@dataclass
class Instance:
    """
    Represents an instance of the RTSM problem.
    """

    variants: List[str]
    tests: List[str]
    performance_matrix: np.ndarray = field(default=None)
    """Matrix (variant, test)"""

    def __post_init__(self):
        self.performance_matrix = np.zeros((len(self.variants), len(self.tests)))

    def store_performance(
        self, variant: Union[str, int], test: Union[str, int], performance: float
    ):
        """
        Store the specified performance.
        """
        vi = self.variants.index(variant) if isinstance(variant, str) else variant
        ti = self.tests.index(test) if isinstance(test, str) else test
        self.performance_matrix[vi, ti] = performance

    def get_tests(self, sol: Tuple[bool, ...]) -> List[str]:
        """
        Convert a boolean tuple mask into the list of selected tests.
        """
        return [test for accept, test in zip(sol, self.tests) if accept]
