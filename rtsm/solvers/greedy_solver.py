from typing import Callable, Set, Tuple, Union

import numpy as np

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver
from rtsm.utils.color_helper import get_color_helper
from rtsm.utils.progress_bar import ProgressBar

F = get_color_helper()


def __new_sol__(
    sol: Tuple[bool, ...],
    current_best: int,
    pbar: ProgressBar,
):
    score = sum(sol)
    if score < current_best:
        pbar.set_best(score, score / len(sol))
        return score
    return current_best


class GreedySolver(Solver):
    """
    A solver that greedily chooses tests.

    """

    def get_name(self) -> str:
        return "greedy"

    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        verbose: bool = False,
        **kwargs,
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
        init = instance.warm_start()
        self.best_sol = {init}
        best_cost = sum(init)
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_cost}{F.RESET} ({F.LIGHTCYAN_EX}{best_cost / len(init):.1%}{F.RESET})"
            )

        pbar = ProgressBar(total=best_cost, name=self.get_name(), use_tqdm=use_tqdm)
        allowed = list(init)
        for _ in range(best_cost):
            mini = float("inf")
            mini_index = 0
            costs = np.sum(instance.performance_matrix[:, :, allowed], axis=(0, 1))
            j = 0
            for x in allowed:
                if x and costs[j] < mini:
                    mini = costs[j]
                    mini_index = j
                j += x
            allowed[mini_index] = False
            if predictor.can_predict(allowed):
                best_cost = __new_sol__(allowed, best_cost, pbar)
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

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()
