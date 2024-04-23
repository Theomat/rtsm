from typing import Tuple

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor, ranking_error, to_ranking

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
    error = np.mean(pred != y)
    return pred, error


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
        X = self.Xt[:, :, usable]
        Y = self.Yt[:, :, [not x for x in usable]]

        for h in range(X.shape[0]):
            for i in range(Y.shape[-1]):
                pred, error = __learn_boolean_linear_model__(
                    X[h, :, :], Y[h, :, i].reshape((-1))
                )
            if error > 0:
                return False
        return True

    def get_ranking(self, usable: Tuple[bool, ...]) -> np.ndarray:
        X = self.Xt[:, :, usable]
        mask = [not x for x in usable]
        Y = self.Yt[:, :, mask]
        cp = self.Xt.copy()

        for h in range(X.shape[0]):
            for i in range(Y.shape[-1]):
                pred, error = __learn_boolean_linear_model__(
                    X[h, :, :], Y[h, :, i].reshape((-1))
                )
                cp[:, :, mask][:, :, i] = pred
        return (to_ranking(np.sum(cp, axis=-1)),)
