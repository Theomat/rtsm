from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, Optional, Set, Tuple, Union, List, Generator

import numpy as np

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver
from rtsm.utils.color_helper import get_color_helper
from rtsm.utils.progress_bar import ProgressBar

F = get_color_helper()


def __new_sol__(
    sol: np.ndarray,
    current_best: int,
    solutions: List[np.ndarray],
    pbar: ProgressBar,
    converter: Callable[[np.ndarray], Solution],
    on_progress_callback: Optional[Callable[[List[Solution]], None]] = None,
):
    score = np.sum(sol)
    if score < current_best:
        pbar.set_best(score, score / len(sol))
        solutions.clear()
        solutions.append(sol.copy())
        if on_progress_callback is not None:
            on_progress_callback([converter(sol)])
        return score
    elif score == current_best:
        solutions.append(sol.copy())
    return current_best


def __sample_candidates__(
    n: int,
    size: int,
    predictor: Predictor,
    seed: Union[int, np.random.Generator],
    max_samples: int,
) -> Generator[np.ndarray, int, None]:
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    choices = list(range(size))
    new_best = size
    sol = np.asarray([False for i in range(n)])
    for used in range(max_samples):
        size_of_sample = rng.integers(1, new_best)
        rng.shuffle(choices)
        selected = choices[:size_of_sample]
        for i in range(size):
            sol[i] = i in selected
        new_best = yield sol


class StupidSolver(Solver):
    """
    A solver that use SWAY to find solutions.

    """

    def get_name(self) -> str:
        return "stupid"

    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        on_progress_callback: Optional[Callable[[Set[Solution]], None]] = None,
        samples: int = 10000,
        seed: Optional[int] = None,
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
        n = len(instance.tests)
        init = np.asarray(instance.warm_start())
        self.best_sol = [init]
        best_cost = np.sum(init)
        initial_cost = best_cost
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_cost}{F.RESET} ({F.LIGHTCYAN_EX}{best_cost / len(init):.1%}{F.RESET})"
            )

        def convert(array):
            return Solution(self.instance, tuple(self.instance.get_tests(array)))

        rng = np.random.default_rng(seed)
        pbar = ProgressBar(total=samples, name=self.get_name(), use_tqdm=use_tqdm)
        pbar.set_best(best_cost, best_cost / len(init))
        gen = __sample_candidates__(n, best_cost, predictor, rng, samples + 1)
        next(gen)
        for _ in range(samples):
            candidate = gen.send(best_cost)
            if predictor.can_predict(candidate):
                best_cost = __new_sol__(
                    candidate,
                    best_cost,
                    self.best_sol,
                    pbar,
                    convert,
                    on_progress_callback,
                )
            pbar.update(1)
            if best_cost == 1:
                break
        pbar.close()

        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return {
            Solution(self.instance, tuple(self.instance.get_tests(sol)))
            for sol in self.best_sol
        }

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()
