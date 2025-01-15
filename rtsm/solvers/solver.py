from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Set, Union

import numpy as np

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.utils.color_helper import get_color_helper

F = get_color_helper()


def get_cost(instance: Instance, other) -> float:
    if isinstance(other, np.ndarray):
        return np.sum(instance.costs[other])
    return Solution(instance, other).cost()


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
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        on_progress_callback: Optional[Callable[[Set[Solution]], None]] = None,
        **kwargs: Any,
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

    def _get_print_prefix_(self) -> str:
        return f"{F.LIGHTYELLOW_EX}{self.get_name()}{F.RESET}:"
