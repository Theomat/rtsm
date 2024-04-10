from abc import ABC, abstractmethod
from typing import Any, Set, Tuple

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution


class Solver(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this solver.
        """
        pass

    @abstractmethod
    def solve(
        self,
        instance: Instance,
        predictor: Predictor,
        use_tqdm: bool = False,
        nprocs: int = 1,
        **kwargs: Any
    ) -> Set[Solution]:
        """
        Try to solve an instance of RTSM and provides a set of solutions.
        """
        pass
