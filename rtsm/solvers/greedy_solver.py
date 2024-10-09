import numpy as np

from rtsm.predictors.predictor import Predictor
from rtsm.solvers.deterministic_solver import DeterministicSolver


class GreedySolver(DeterministicSolver):
    """
    A solver that greedily chooses tests.

    """

    def get_name(self) -> str:
        return "greedy"

    def __choose_index_to_remove__(
        self, current: np.ndarray, predictor: Predictor
    ) -> int:
        mini = float("inf")
        mini_index = 0
        costs = np.sum(self.instance.performance_matrix[:, :, current], axis=(0, 1))
        j = 0
        for x in current:
            if x and costs[j] < mini:
                mini = costs[j]
                mini_index = j
            j += x
        return mini_index
