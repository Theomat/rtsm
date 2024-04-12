from typing import Tuple

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor

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
    error = np.mean(model.predict(A) != y)
    return model.coef_, error


class LogisticBooleanPredictor(Predictor):
    """
    A predictor that predicts boolean results using logistic regression.
    """

    def __init__(self, instance: Instance, **kwargs) -> None:
        self.instance = instance
        self.Xt = instance.performance_matrix.copy()
        self.Yt = self.Xt.copy()

    def get_name(self) -> str:
        return "logistic-boolean"

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        X = self.Xt[:, usable]
        Y = self.Yt[:, [not x for x in usable]]

        for i in range(Y.shape[1]):
            coeffs, error = __learn_boolean_linear_model__(X, Y[:, i].reshape((-1)))
            if error > 0:
                return False
        return True
