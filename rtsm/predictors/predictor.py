from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from rtsm.instance import Instance


def to_ranking(perfs: np.ndarray) -> np.ndarray:
    """
    Transforms a performance matrix (k, variant) into a ranking matrix.
    """
    n = perfs.shape[1]
    return np.repeat(perfs, n, axis=1).reshape((-1, n, n)) > perfs.reshape((-1, 1, n))


def ranking_error(target_ranks: np.ndarray, pred_ranks: np.ndarray) -> float:
    """
    Takes two ranking matrices and returns the percentage of errors.
    The percentage of errors is percentage of order relations that were different.
    """
    num = np.max(np.sum(target_ranks != pred_ranks, axis=(1, 2)))
    n = target_ranks.shape[1]
    return num / (np.prod(target_ranks.shape[1:]) - n)


class Predictor(ABC):
    def __init__(self, instance: Instance, **kwargs) -> None:
        self.instance = instance

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this solver.
        """
        pass

    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        """
        Return true if and only if the performances are predictable according to this predictor.
        """
        return self.ranking_error(usable) <= 0

    @abstractmethod
    def ranking_error(self, usable: Tuple[bool, ...]) -> float:
        """
        Compute the ranking error after we learn a predicotr given the usable information.
        """
        pass

    def export_prediction(self, usable: Tuple[bool, ...], path: str) -> None:
        """
        Export the prediction for the specified mask to the specified path.
        """
        raise NotImplementedError()
