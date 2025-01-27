from typing import List, Tuple
import json

from rtsm.instance import Instance
from rtsm.predictors.predictor import Prediction, Predictor, to_ranking, ranking_error

import numpy as np


class NoPrediction(Prediction):
    def __init__(self, inputs: List[str], outputs: List[str]) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def predict(self, information: np.ndarray) -> np.ndarray:
        out = np.sum(information, axis=-1)
        return out

    def export(self, path: str) -> None:
        out = {
            "type": "none",
            "input": self.inputs,
            "output": self.outputs,
        }
        with open(path, "w") as fd:
            json.dump(out, fd)


class NoPredictor(Predictor):
    """
    A predictor that does not predict anything.
    """

    def __init__(self, instance: Instance, accuracy: float = 1.0, **kwargs) -> None:
        self.instance = instance
        self.Xt = instance.performance_matrix.copy()
        self.Rt = to_ranking(np.sum(self.Xt, axis=-1))
        self.accuracy = accuracy

    def get_name(self) -> str:
        return "none"

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        return ranking_error(self.Rt, self.get_ranking(usable)) <= 1 - self.accuracy

    def get_ranking(self, usable: Tuple[bool, ...]) -> np.ndarray:
        X = self.Xt[:, :, usable]
        D = np.sum(X, axis=-1)
        return to_ranking(D)

    def export_prediction(self, usable: Tuple[bool]):
        return NoPrediction(
            self.instance.get_tests(usable),
            [t for t, b in zip(self.instance.tests, usable) if not b],
        )
