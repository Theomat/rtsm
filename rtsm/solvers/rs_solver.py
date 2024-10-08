from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, Optional, Set, Tuple, Union, List

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
    print(score, current_best, sol, solutions)
    if score < current_best:
        pbar.set_best(score, score / len(sol))
        solutions.clear()
        solutions.append(sol)
        if on_progress_callback is not None:
            on_progress_callback([converter(sol)])
        return score
    elif score == current_best:
        solutions.append(sol)
    return current_best


def __sample__(
    n: int,
    size: int,
    predictor: Predictor,
    seed: Union[int, np.random.Generator],
    max_samples: int,
) -> Tuple[bool, int, np.ndarray]:
    choices = list(range(size))
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    for used in range(max_samples):
        selected = rng.choice(choices, n, replace=False)
        current = np.asarray([i in selected for i in range(size)])
        if predictor.can_predict(current):
            return True, used + 1, current
    return False, max_samples, None


class RandomSolutionSolver(Solver):
    """
    A solver that samples random solutions.

    """

    def get_name(self) -> str:
        return "rs"

    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        samples: int = 10000,
        seed: Optional[int] = None,
        verbose: bool = False,
        on_progress_callback: Optional[Callable[[Set[Solution]], None]] = None,
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
        SAMPLING_UNIT = 100
        n = len(instance.tests)
        init = np.asarray(instance.warm_start())
        self.best_sol = [init]
        best_cost = np.sum(init)
        initial_cost = best_cost
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_cost}{F.RESET} ({F.LIGHTCYAN_EX}{best_cost / len(init):.1%}{F.RESET})"
            )

        budget = samples

        def convert(array):
            return Solution(self.instance, tuple(self.instance.get_tests(array)))

        pbar = ProgressBar(total=samples, name=self.get_name(), use_tqdm=use_tqdm)
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            # Find best among possible children
            queued = seed or 0
            while budget > 0 and best_cost > 1:
                while len(futures) < nprocs:
                    futures.append(
                        pool.submit(
                            __sample__,
                            initial_cost - 1,
                            n,
                            predictor,
                            queued,
                            min(SAMPLING_UNIT, budget),
                        )
                    )
                    queued += 1
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    has_found, used, out = future.result()
                    futures.remove(future)
                    budget -= used
                    pbar.update(used)
                    if not has_found:
                        continue
                    best_cost = __new_sol__(
                        out,
                        best_cost,
                        self.best_sol,
                        pbar,
                        convert,
                        on_progress_callback,
                    )
            pool.shutdown()

        else:
            while budget > 0 and best_cost > 1:
                has_found, used, out = __sample__(
                    initial_cost - 1,
                    n,
                    predictor,
                    (seed or 0) + budget,
                    min(SAMPLING_UNIT, budget),
                )
                budget -= used
                pbar.update(used)
                if not has_found:
                    continue
                best_cost = __new_sol__(
                    out, best_cost, self.best_sol, pbar, convert, on_progress_callback
                )
        pbar.close()

        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return {
            Solution(self.instance, tuple(self.instance.get_tests(sol)))
            for sol in self.best_sol
        }

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()
