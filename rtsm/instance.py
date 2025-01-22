from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class Instance:
    """
    Represents an instance of the RTSM problem.
    """

    performances: List[str]
    variants: List[str]
    tests: List[str]
    performance_matrix: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    weights: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    costs: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    """Matrix (performance, variant, test)"""
    __check: Optional[np.ndarray] = field(
        default=None, compare=False, repr=False, hash=False
    )
    __warm_start: Optional[Tuple[bool, ...]] = field(
        default=None, compare=False, repr=False, hash=False
    )

    def __post_init__(self):
        self.performances.sort()
        self.variants.sort()
        self.tests.sort()
        self.performance_matrix = np.zeros(
            (len(self.performances), len(self.variants), len(self.tests))
        )
        self.weights = np.zeros((len(self.tests),))
        self.costs = np.zeros((len(self.tests),))
        self.__check = np.zeros_like(self.performance_matrix)

    def total_cost(self) -> float:
        return np.sum(self.costs)

    def copy(self) -> "Instance":
        """
        Makes a shallow copy of this instance.
        """
        i = Instance(self.variants, self.tests)
        i.performance_matrix = self.performance_matrix
        i.weights = self.weights
        i.costs = self.costs
        i.__check = self.__check
        i.__warm_start = self.__warm_start
        return i

    def set_start(self, tests: List[str]) -> None:
        """
        Set the initial start of this instance to the selected subset of tests.
        """
        self.__warm_start = tuple(t in tests for t in self.tests)

    def warm_start(self) -> Tuple[bool, ...]:
        """
        Gives an initial start to solve this instance.
        It is guaranteed that the warm start satisfies the solving constraints.
        """
        if self.__warm_start is None:
            self.__warm_start = tuple(True for _ in range(len(self.tests)))
        return self.__warm_start

    def swap(self) -> "Instance":
        """
        Return the same instance but with variants and tests swapped.
        """
        instance = Instance(self.performances[:], self.tests[:], self.variants[:])
        for h, performance in enumerate(self.performances):
            for i, variant in enumerate(self.variants):
                for index, test in enumerate(self.tests):
                    instance.store_performance(
                        h, index, i, self.performance_matrix[h, i, index]
                    )
        return instance

    def subset(self, selected_tests: List[str]) -> "Instance":
        """
        Returns the instance where only the selected subset of tests are kept.
        """
        index2test = {self.tests.index(t): t for t in selected_tests}
        sorted_tests = [index2test[i] for i in sorted(index2test.keys())]
        instance = Instance(self.performances, self.variants, sorted_tests)
        for h, performance in enumerate(self.performances):
            for i, variant in enumerate(self.variants):
                for index, test in index2test.items():
                    instance.store_performance(
                        h, i, test, self.performance_matrix[h, i, index]
                    )
        for index, test in index2test.items():
            instance.store_cost(test, self.costs[index])
        # mask = [t in selected_tests for t in self.tests]
        # assert np.allclose(self.performance_matrix[:, :, mask], instance.performance_matrix)
        if self.__warm_start is not None:
            start = [t for t, b in zip(self.tests, self.warm_start()) if b]
            instance.set_start(start)
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
            if len(test_i[start:end]) == 0:
                parts -= 1
                start = end
                continue
            out.append(self.subset(test_i[start:end]))
            start = end
            parts -= 1
        return out

    def store_performance(
        self,
        performance: Union[str, int],
        variant: Union[str, int],
        test: Union[str, int],
        value: float,
    ):
        """
        Store the specified performance.
        """
        pi = (
            self.performances.index(performance)
            if isinstance(performance, str)
            else performance
        )
        vi = self.variants.index(variant) if isinstance(variant, str) else variant
        ti = self.tests.index(test) if isinstance(test, str) else test
        self.performance_matrix[pi, vi, ti] = value
        if self.__check is not None and self.__check[pi, vi, ti] <= 0:
            self.__check[pi, vi, ti] = 1

    def store_cost(
        self,
        test: Union[str, int],
        value: float,
    ):
        ti = self.tests.index(test) if isinstance(test, str) else test
        self.costs[ti] = value

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
            self.weights = np.exp(-self.costs) / np.sum(np.exp(-self.costs))
        return filled

    def get_tests(self, sol: Tuple[bool, ...]) -> List[str]:
        """
        Convert a boolean tuple mask into the list of selected tests.
        """
        return [test for accept, test in zip(sol, self.tests) if accept]
