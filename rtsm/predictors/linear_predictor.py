from typing import Tuple

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor, to_ranking, ranking_error

import numpy as np
from sklearn.linear_model import LinearRegression


def __learn_linear_model__(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = LinearRegression()
    model.fit(A, y)
    return model.predict(A)


class LinearRegressionPredictor(Predictor):
    """
    A predictor that predicts float values using linear regression.
    """

    def __init__(self, instance: Instance, accuracy: float = 1.0, **kwargs) -> None:
        self.instance = instance
        self.Xt = instance.performance_matrix.copy()
        self.Yt = self.Xt.copy()
        self.D: np.ndarray = np.sum(self.Xt, axis=-1)
        self.Rt = to_ranking(self.D)
        self.accuracy = accuracy

    def get_name(self) -> str:
        return "linear"

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        X = self.Xt[:, usable]
        Y = self.Yt[:, [not x for x in usable]]
        D = self.D.copy()
        for i in range(Y.shape[1]):
            new_row = __learn_linear_model__(X, Y[:, i].reshape((-1)))
            D += new_row - Y[:, i]
            if ranking_error(self.Rt, to_ranking(D)) > 1 - self.accuracy:
                return False
        return True
