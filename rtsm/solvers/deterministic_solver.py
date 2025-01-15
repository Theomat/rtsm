from typing import Callable, Optional, Set, Union, Any

import numpy as np

from abc import abstractmethod, ABC

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver, get_cost
from rtsm.utils.color_helper import get_color_helper
from rtsm.utils.progress_bar import ProgressBar

F = get_color_helper()


def __new_sol__(
    sol: np.ndarray, current_best: int, pbar: ProgressBar, instance: Instance
):
    score = get_cost(instance, sol)
    if score < current_best:
        pbar.set_best(score, score / instance.total_cost())
        return score
    return current_best


class DeterministicSolver(Solver, ABC):
    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        on_progress_callback: Optional[Callable[[Set[Solution]], None]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Set[Solution]:
        """
        Try to solve an instance of RTSM and provides a set of solutions.
        """
        predictor = (
            predictor_builder(instance)
            if not isinstance(predictor_builder, Predictor)
            else predictor_builder
        )
        self.instance = instance
        init = np.asarray(instance.warm_start())
        self.best_sol = [init]
        best_cost = get_cost(instance, init)
        total_cost = instance.total_cost()
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_cost}{F.RESET} ({F.LIGHTCYAN_EX}{best_cost / total_cost:.1%}{F.RESET})"
            )

        pbar = ProgressBar(total=np.sum(init), name=self.get_name(), use_tqdm=use_tqdm)
        allowed = np.copy(init)
        for _ in range(best_cost):
            to_remove_index = self.__choose_index_to_remove__(allowed, predictor)
            if to_remove_index is None:
                break
            allowed[to_remove_index] = False
            if predictor.can_predict(allowed):
                best_cost = __new_sol__(allowed, best_cost, pbar, instance)
                self.best_sol = [allowed]
            else:
                break
            pbar.update(1)
        pbar.close()

        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return {
            Solution(self.instance, tuple(self.instance.get_tests(sol)))
            for sol in self.best_sol
        }

    @abstractmethod
    def __choose_index_to_remove__(
        self, current: np.ndarray, predictor: Predictor
    ) -> Optional[int]:
        pass

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()
