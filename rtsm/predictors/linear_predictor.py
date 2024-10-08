from typing import Tuple, List
import json

from rtsm.instance import Instance
from rtsm.predictors.predictor import (
    Predictor,
    Prediction,
    to_ranking,
    to_ranking1d,
    ranking_error,
    ranking_error2d,
)

import numpy as np
from sklearn.linear_model import LinearRegression


def __learn_linear_model__(
    A: np.ndarray, y: np.ndarray, positive: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = LinearRegression(positive=positive)
    try:
        model.fit(A, y)
    except RuntimeError:
        # Max iterations reached
        return np.zeros((A.shape[1],)), np.median(y), np.median(y)
    return model.coef_, model.intercept_, model.predict(A)


class LinearPrediction(Prediction):
    def __init__(
        self, inputs: List[str], outputs: List[str], A: np.ndarray, b: np.ndarray
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.A = A
        self.b = b

    def predict(self, information: np.ndarray) -> np.ndarray:
        p = self.A.shape[0]
        n = self.A.shape[1]
        variants = information.shape[1]
        information = information.transpose((0, 2, 1))
        out = np.dot(self.A, information).reshape((p, n, variants)).transpose((0, 2, 1))
        out += self.b
        return out

    def export(self, path: str) -> None:
        out = {
            "type": "linear",
            "input": self.inputs,
            "output": self.outputs,
            "coefficients": self.A.tolist(),
            "translation": self.b.tolist(),
        }
        with open(path, "w") as fd:
            json.dump(out, fd)


class LinearRegressionPredictor(Predictor):
    """
    A predictor that predicts float values using linear regression.
    """

    def __init__(
        self,
        instance: Instance,
        accuracy: float = 1.0,
        positive: bool = False,
        **kwargs,
    ) -> None:
        self.instance = instance
        self.Xt = instance.performance_matrix.copy()
        self.Yt = self.Xt.copy()
        self.Rt = to_ranking(np.sum(self.Xt, axis=-1))
        self.accuracy = accuracy
        self.positive = positive

    def get_name(self) -> str:
        if self.positive:
            return "linear+"
        return "linear"

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        mask = np.asarray(usable)
        X = self.Xt[:, :, mask]
        Y = self.Yt[:, :, ~mask]
        for h in range(X.shape[0]):
            D = np.add.reduce(self.Xt[h, :, :], axis=-1)
            for i in range(Y.shape[-1]):
                D[:] += __learn_linear_model__(
                    X[h, :, :], Y[h, :, i].reshape((-1)), self.positive
                )[-1]
            if ranking_error2d(self.Rt[h, :, :], to_ranking1d(D)) > 1 - self.accuracy:
                return False
        return True

    def get_ranking(self, usable: Tuple[bool, ...]) -> np.ndarray:
        mask = np.asarray(usable)
        X = self.Xt[:, :, mask]
        Y = self.Yt[:, :, ~mask]
        D = np.add.reduce(self.Xt, axis=-1)
        for h in range(X.shape[0]):
            for i in range(Y.shape[-1]):
                D[h, :] += __learn_linear_model__(
                    X[h, :, :], Y[h, :, i].reshape((-1)), self.positive
                )[-1]
        return to_ranking(D)

    def export_prediction(self, usable: Tuple[bool]) -> LinearPrediction:
        mask = np.asarray(usable)
        X = self.Xt[:, :, mask]
        Y = self.Yt[:, :, ~mask]
        coeffs = np.zeros((X.shape[0], X.shape[-1], Y.shape[-1]))
        intercepts = np.zeros((X.shape[0], Y.shape[-1]))
        for h in range(X.shape[0]):
            for i in range(Y.shape[-1]):
                coeff, intercept, _ = __learn_linear_model__(
                    X[h, :, :], Y[h, :, i].reshape((-1)), self.positive
                )
                coeffs[h, :, i] = coeff
                intercepts[h, i] = intercept

        return LinearPrediction(
            self.instance.get_tests(usable),
            [t for t, b in zip(self.instance.tests, usable) if not b],
            coeffs.transpose((0, 2, 1)),
            intercepts,
        )
