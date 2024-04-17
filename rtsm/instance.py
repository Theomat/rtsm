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

    def warm_start(self) -> Tuple[bool, ...]:
        """
        Compute a warm start for this instance.
        A warm start is an initial value which does not prevent optimality but remove useless data.
        This works well with boolean data.
        """
        classes = set()
        selected = [True for _ in range(len(self.tests))]
        for i, row in enumerate(self.performance_matrix):
            key = tuple(x for x in row)
            if key in classes:
                selected[i] = False
            else:
                classes.add(key)
        return tuple(selected)

    def swap(self) -> "Instance":
        """
        Return the same instance but with variants and tests swapped.
        """
        i = Instance(self.tests, self.variants)
        i.performance_matrix = self.performance_matrix.transpose()
        i.__check = self.__check.transpose() if self.__check is not None else None
        return i

    def subset(self, selected_tests: List[str]) -> "Instance":
        """
        Returns the instance where only the selected subset of tests are kept.
        """
        index2test = {self.tests.index(t): t for t in selected_tests}
        instance = Instance(self.variants[:], list(index2test.values()))
        for i, variant in enumerate(self.variants):
            for index, test in index2test.items():
                instance.store_performance(
                    variant, test, self.performance_matrix[i, index]
                )
        return instance

    def split(self, n: int, seed: Optional[int] = None) -> List["Instance"]:
        """
        Split this instance into n instances, the variants are kepts but random subsets of tests are used.
        """
        rng = np.random.default_rng(seed)
        test_i = self.tests[:]
        rng.shuffle(test_i)
        parts = n
        size = len(self.tests) // n
        out = []
        start = 0
        while parts > 0:
            end = start + size
            if parts == 1:
                end = len(self.tests)
            out.append(self.subset(test_i[start:end]))
            start = end
            parts -= 1
        return out

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
