from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, Optional, Set, Tuple, Union

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
    solutions: Set[Tuple[bool, ...]],
    pbar: ProgressBar,
):
    score = sum(sol)
    if score < current_best:
        pbar.set_best(score, score / len(sol))
        solutions.clear()
        solutions.add(sol)
        return score
    elif score == current_best and sol not in solutions:
        solutions.add(sol)
    return current_best


def __sample__(
    current_sol: Optional[Tuple[bool, ...]],
    n: int,
    size: int,
    predictor: Predictor,
    seed: Union[int, np.random.Generator],
    max_samples: int,
) -> Tuple[bool, int, Tuple[bool, ...]]:
    choices = (
        list(range(size))
        if current_sol is None
        else [i for i, v in enumerate(current_sol) if v]
    )
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    done = set()
    for used in range(max_samples):
        selected = rng.choice(choices, n, replace=False)
        key = tuple(selected)
        if key in done:
            continue
        done.add(key)
        current = [i in selected for i in range(size)]
        if predictor.can_predict(current):
            return True, used + 1, tuple(current)
    return False, max_samples, tuple()


class RandomImprovementSolver(Solver):
    """
    A solver that samples random improvements.

    """

    def __init__(self, local: bool = False) -> None:
        super().__init__()
        self.local = local

    def get_name(self) -> str:
        return "ri" + ("-local" if self.local else "")

    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
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
        SAMPLING_UNIT = 100
        n = len(instance.tests)
        init = instance.warm_start()
        self.best_sol = {init}
        best_cost = sum(init)
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_cost}{F.RESET} ({F.LIGHTCYAN_EX}{best_cost / len(init):.1%}{F.RESET})"
            )

        budget = samples
        current_sol = init if self.local else None

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
                            current_sol,
                            best_cost - 1,
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
                    )
                    current_sol = list(self.best_sol)[0] if self.local else None
            pool.shutdown()

        else:
            while budget > 0 and best_cost > 1:
                has_found, used, out = __sample__(
                    current_sol,
                    best_cost - 1,
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
                    out,
                    best_cost,
                    self.best_sol,
                    pbar,
                )
                current_sol = list(self.best_sol)[0] if self.local else None

        pbar.close()

        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return {
            Solution(self.instance, tuple(self.instance.get_tests(sol)))
            for sol in self.best_sol
        }

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()
