from typing import Tuple

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor, to_ranking, ranking_error

import numpy as np
from sklearn.linear_model import LogisticRegression


def __learn_boolean_linear_model__(
    A: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, float]:
    if np.all(y == 1):
        return 1, 0
    elif np.all(y == 0):
        return 0, 0
    model = LogisticRegression(solver="liblinear", penalty="l1")
    model.fit(A, y)
    pred = model.predict(A)
    error = np.sum(pred != y)
    return pred, error


class LogisticRankPredictor(Predictor):
    """
    A predictor that predicts relative ranks using a logistic model.
    """

    def __init__(self, instance: Instance, accuracy: float = 1.0, **kwargs) -> None:
        self.instance = instance
        m = instance.performance_matrix.shape[0]
        n = instance.performance_matrix.shape[1]
        self.Rt = to_ranking(np.sum(self.instance.performance_matrix, axis=-1))
        self.Xt = np.zeros((m * m, 2 * n))
        for i in range(m):
            self.Xt[i * m : (i + 1) * m, n:] = instance.performance_matrix[:, :]
            self.Xt[i * m : (i + 1) * m, :n] = np.broadcast_to(
                instance.performance_matrix[i, :], (m, n)
            )

        self.accuracy = accuracy

    def get_name(self) -> str:
        return "logistic-rank"

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        mask = usable + usable
        X = self.Xt[:, mask]
        new_R = np.zeros_like(self.Rt)
        total_error = 0
        m = self.instance.performance_matrix.shape[0]
        n = self.instance.performance_matrix.shape[1]
        for i in range(m):
            pred, error = __learn_boolean_linear_model__(
                X[i * m : (i + 1) * m, :], self.Rt[i, :]
            )
            new_R[i, :] = pred
            total_error += error
            if total_error / (n * n) > 1 - self.accuracy:
                return False

        if ranking_error(self.Rt, new_R) > 1 - self.accuracy:
            return False
        return True
