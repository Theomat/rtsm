import numpy as np

from typing import Optional

from rtsm.predictors.predictor import Predictor
from rtsm.solvers.deterministic_solver import DeterministicSolver

from scipy.stats import rankdata, t, friedmanchisquare


class FriedmanSolver(DeterministicSolver):
    """
    A solver that Friedman chooses tests.
    """

    def get_name(self) -> str:
        return "friedman"

    def __choose_index_to_remove__(
        self, current: np.ndarray, predictor: Predictor
    ) -> Optional[int]:
        alpha = 0.05
        X = self.instance.performance_matrix[:, :, current]
        X = X.reshape((-1, np.sum(current))).T
        full_ranks = rankdata(X, axis=-1)
        ranks = np.add.reduce(full_ranks, axis=-1)
        n = X.shape[-1]
        k = X.shape[0]
        T, p = friedmanchisquare(*[X[i] for i in range(n)])
        if p > alpha:
            return None
        # print(n, k)
        # print(np.max(full_ranks))
        student = t.ppf(1 - alpha / 2, n)
        c1 = np.sum(np.square(full_ranks)) - k * n * (n + 1) ** 2 / 4
        c2 = 1 - T / (k * (n - 1))
        divider = np.sqrt(2 * k * c2 * c1 / ((k - 1) * (n - 1)))
        # print(
        #     c1, c2, divider, np.sum(np.square(full_ranks)), k * n * ((n + 1) ** 2) / 4
        # )
        best = np.argmin(ranks)
        rj = ranks[best]
        for h in range(n):
            if h == best:
                continue
            rh = ranks[h]
            if np.abs(rj - rh) / divider > student:
                return h
        return None
