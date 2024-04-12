from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class Instance:
    """
    Represents an instance of the RTSM problem.
    """

    variants: List[str]
    tests: List[str]
    performance_matrix: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    """Matrix (variant, test)"""
    __check: Optional[np.ndarray] = field(
        default=None, compare=False, repr=False, hash=False
    )

    def __post_init__(self):
        self.performance_matrix = np.zeros((len(self.variants), len(self.tests)))
        self.__check = np.zeros_like(self.performance_matrix)

    def swap(self) -> "Instance":
        """
        Return the same instance but with variants and tests swapped.
        """
        i = Instance(self.tests, self.variants)
        i.performance_matrix = self.performance_matrix.transpose()
        i.__check = self.__check.transpose() if self.__check is not None else None
        return i

    def store_performance(
        self, variant: Union[str, int], test: Union[str, int], performance: float
    ):
        """
        Store the specified performance.
        """
        vi = self.variants.index(variant) if isinstance(variant, str) else variant
        ti = self.tests.index(test) if isinstance(test, str) else test
        self.performance_matrix[vi, ti] = performance
        if self.__check is not None:
            self.__check[vi, ti] = 1

    def check_filled(self) -> bool:
        """
        Check that the performance matrix has been filled.
        """
        if self.__check is None:
            return True
        total = np.sum(self.__check) / np.prod(self.__check.shape)
        filled = total >= 1
        if filled:
            self.__check = None
        return filled

    def get_tests(self, sol: Tuple[bool, ...]) -> List[str]:
        """
        Convert a boolean tuple mask into the list of selected tests.
        """
        return [test for accept, test in zip(sol, self.tests) if accept]
