from typing import Tuple, List

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor, to_ranking, ranking_error

import numpy as np


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
