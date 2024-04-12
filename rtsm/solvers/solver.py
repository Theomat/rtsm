from abc import ABC, abstractmethod
from typing import Any, Set

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

    @abstractmethod
    def early_exit(self) -> Set[Solution]:
        """
        Called when when the solving is exited before solve has finished.
        All resources should be freed and the current solution should be returned.
        """
        pass
