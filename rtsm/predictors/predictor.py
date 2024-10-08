from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple

import numpy as np

from rtsm.instance import Instance


def to_ranking(perfs: np.ndarray) -> np.ndarray:
    """
    Transforms a performance matrix (k, variant) into a ranking matrix.
    """
    n = perfs.shape[1]
    return np.repeat(perfs, n, axis=1).reshape((-1, n, n)) > perfs.reshape((-1, 1, n))


def to_ranking1d(perfs: np.ndarray) -> np.ndarray:
    """
    Transforms a performance matrix (variant) into a ranking matrix.
    """
    return np.greater.outer(perfs, perfs)


def ranking_error(target_ranks: np.ndarray, pred_ranks: np.ndarray) -> float:
    """
    Takes two ranking matrices and returns the percentage of errors.
    The percentage of errors is percentage of order relations that were different.
    """
    num = np.max(np.add.reduce(target_ranks != pred_ranks, axis=(1, 2)))
    n = target_ranks.shape[1]
    return num / (np.prod(target_ranks.shape[1:]) - n)


def ranking_error2d(target_ranks: np.ndarray, pred_ranks: np.ndarray) -> float:
    """
    Takes two ranking matrices and returns the percentage of errors.
    The percentage of errors is percentage of order relations that were different.
    """
    num = np.add.reduce(target_ranks != pred_ranks, axis=(0, 1))
    n = target_ranks.shape[1]
    return num / (np.prod(target_ranks.shape) - n)


def to_ranks(ranking_matrix: np.ndarray) -> np.ndarray:
    """
    Return a matrix with ranks of each variant as integers.
    """
    p = ranking_matrix.shape[0]
    n = ranking_matrix.shape[1]
    ranks = np.zeros((p, n))
    for h in range(p):
        dico = defaultdict(list)
        for i in range(n):
            pot_rank = n - np.sum(ranking_matrix[h, i])
            dico[pot_rank].append(i)

        current_rank = 0
        for i in range(n + 1):
            candidates = dico[i]
            if len(candidates) >= 1:
                if len(candidates) > 1:
                    ordered = []
                    for candidate in candidates:
                        ordered.append(
                            (np.sum(ranking_matrix[h, candidates]), candidate)
                        )
                    new_rank = sorted(ordered, reverse=True)
                    for _, el in new_rank:
                        ranks[h, el] = current_rank
                        current_rank += 1
                else:
                    ranks[h, candidates[0]] = current_rank
                    current_rank += 1
    return ranks


class Prediction(ABC):
    @abstractmethod
    def predict(self, information: np.ndarray) -> np.ndarray:
        """
        Uses the prediction object to predict.
        """
        pass

    def export(self, path: str) -> None:
        """
        Export this prediction to the specified file.
        """
        raise NotImplementedError()


class Predictor(ABC):
    def __init__(self, instance: Instance, **kwargs) -> None:
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

    @abstractmethod
    def get_ranking(self, usable: Tuple[bool, ...]) -> np.ndarray:
        """
        Compute the ranking matrix after we learn a predictor given the usable information.
        """
        pass

    def export_prediction(self, usable: Tuple[bool, ...]) -> Prediction:
        """
        Export the prediction for the specified mask.
        """
        raise NotImplementedError()
