from typing import Tuple
import json

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor, to_ranking, ranking_error

import numpy as np
from sklearn.linear_model import LinearRegression


def __learn_linear_model__(
    A: np.ndarray, y: np.ndarray
) -> Tuple[LinearRegression, np.ndarray]:
    model = LinearRegression()
    model.fit(A, y)
    return model, model.predict(A)


class LinearRegressionPredictor(Predictor):
    """
    A predictor that predicts float values using linear regression.
    """

    def __init__(self, instance: Instance, accuracy: float = 1.0, **kwargs) -> None:
        self.instance = instance
        self.Xt = instance.performance_matrix.copy()
        self.Yt = self.Xt.copy()
        self.Rt = to_ranking(np.sum(self.Xt, axis=-1))
        self.accuracy = accuracy

    def get_name(self) -> str:
        return "linear"

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        return self.ranking_error(usable) <= 1 - self.accuracy

    def ranking_error(self, usable: Tuple[bool, ...]) -> bool:
        X = self.Xt[:, usable]
        Y = self.Yt[:, [not x for x in usable]]
        D = np.sum(self.Xt, axis=-1)
        for i in range(Y.shape[1]):
            D += __learn_linear_model__(X, Y[:, i].reshape((-1)))[1]
        return ranking_error(self.Rt, to_ranking(D))

    def export_prediction(self, usable: Tuple[bool], path: str) -> None:
        out = {
            "type": "linear",
            "input": self.instance.get_tests(usable),
            "output": [t for t, b in zip(self.instance.tests, usable) if not b],
            "coefficients": [],
            "translation": [],
        }
        X = self.Xt[:, usable]
        Y = self.Yt[:, [not x for x in usable]]
        coeffs = np.zeros((X.shape[1], Y.shape[1]))
        intercept = np.zeros((Y.shape[1]))
        for i in range(Y.shape[1]):
            model = __learn_linear_model__(X, Y[:, i].reshape((-1)))[0]
            coeffs[:, i] = model.coef_
            intercept[i] = model.intercept_
        out["coefficients"] = coeffs.tolist()
        out["translation"] = intercept.tolist()
        with open(path, "w") as fd:
            json.dump(out, fd)
