from typing import Tuple, List
import json

from rtsm.instance import Instance
from rtsm.predictors.predictor import (
    Predictor,
    Prediction,
    to_ranking,
    to_ranking1d,
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


class WeightedPrediction(Prediction):
    def __init__(self, inputs: List[str], outputs: List[str], A: np.ndarray) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.A = A

    def predict(self, information: np.ndarray) -> np.ndarray:
        out = np.sum(
            information * self.A.reshape((information.shape[0], 1, -1)), axis=-1
        )
        return out

    def export(self, path: str) -> None:
        out = {
            "type": "weighted",
            "input": self.inputs,
            "output": self.outputs,
            "coefficients": self.A.tolist(),
        }
        with open(path, "w") as fd:
            json.dump(out, fd)


class WeightedPredictor(Predictor):
    """
    A predictor that gives coefficients to each instance.
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
        self.T = np.sum(self.Xt, axis=-1)
        self.Rt = to_ranking(self.T)
        self.accuracy = accuracy
        self.positive = positive

    def get_name(self) -> str:
        if self.positive:
            return "weighted+"
        return "weighted"

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        mask = np.asarray(usable)
        X = self.Xt[:, :, mask]
        if X.shape[-1] <= 1:
            return True
        for h in range(X.shape[0]):
            # D has shape (variants,)
            D = np.add.reduce(X[h], axis=-1)
            learnt = __learn_linear_model__(X[h, :, :], self.T[h], self.positive)[-1]
            D += learnt
            if ranking_error2d(self.Rt[h, :, :], to_ranking1d(D)) > 1 - self.accuracy:
                return False
        return True

    def get_ranking(self, usable: Tuple[bool, ...]) -> np.ndarray:
        mask = np.asarray(usable)
        X = self.Xt[:, :, mask]
        D = np.add.reduce(X, axis=-1)
        for h in range(X.shape[0]):
            learnt = __learn_linear_model__(X[h, :, :], self.T[h], self.positive)[-1]
            D[h, :] += learnt
        return to_ranking(D)

    def export_prediction(self, usable: Tuple[bool]):
        mask = np.asarray(usable)
        X = self.Xt[:, :, mask]
        coeffs = np.zeros(
            (
                X.shape[0],
                X.shape[-1],
            )
        )
        for h in range(X.shape[0]):
            coeff, _, __ = __learn_linear_model__(X[h, :, :], self.T[h], self.positive)
            coeffs[h, :] = coeff

        return WeightedPrediction(
            self.instance.get_tests(usable),
            [t for t, b in zip(self.instance.tests, usable) if not b],
            coeffs,
        )
