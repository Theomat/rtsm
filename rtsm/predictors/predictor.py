from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from rtsm.instance import Instance


def to_ranking(perfs: np.ndarray) -> np.ndarray:
    """
    Transforms a performance vector (variant) into a ranking matrix.
    """
    n = perfs.shape[0]
    return np.broadcast_to(perfs, (n, n)).T > perfs


def ranking_error(target_ranks: np.ndarray, pred_ranks: np.ndarray) -> float:
    """
    Takes two ranking matrices and returns the percentage of errors.
    The percentage of errors is percentage of order relations that were different.
    """
    n = target_ranks.shape[0]
    num = np.sum(target_ranks != pred_ranks) - n
    if num <= 0:
        return 0
    return num / (np.prod(target_ranks.shape) - n)


class Predictor(ABC):
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this solver.
        """
        pass

    @abstractmethod
    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        """
        Return true if and only if the performances are predictable according to this predictor.
        """
        pass
