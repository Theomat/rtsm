from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver
from rtsm.utils.color_helper import get_color_helper
from rtsm.utils.progress_bar import ProgressBar

F = get_color_helper()


def __improve_upon__(
    sol_set: Set[Tuple[bool, ...],],
    seed: Union[int, np.random.Generator],
    predictor: Predictor,
    max_samples: int,
) -> Tuple[bool, ...]:
    one_sol = list(sol_set)[0]
    current = tuple(any(sol[i] for sol in sol_set) for i in range(len(one_sol)))
    must_keep = tuple(all(sol[i] for sol in sol_set) for i in range(len(one_sol)))
    return __sample_specific__(current, must_keep, seed, predictor, max_samples)


def __new_sol__(
    sol: Tuple[bool, ...],
    current_best: int,
    solutions: Set[Tuple[bool, ...]],
    improvement_queue: List[Set[Tuple[bool, ...]]],
    pbar: ProgressBar,
    try_improve: bool,
):
    score = sum(sol)
    if score < current_best:
        pbar.set_best(score, score / len(sol))
        solutions.clear()
        solutions.add(sol)
        return score
    elif score == current_best and sol not in solutions:
        if try_improve:
            for x in solutions:
                improvement_queue.append({x, sol})
        solutions.add(sol)
    return current_best


def __sample_specific__(
    current: Tuple[bool],
    must_keep: Tuple[bool],
    seed: Union[int, np.random.Generator],
    predictor: Predictor,
    max_samples: int,
) -> Tuple[bool, int, Tuple[bool, ...]]:
    size = len(current)
    choices = [i for i in range(size) if current[i] and not must_keep[i]]
    n = len(choices) - 1
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    for used in range(max_samples):
        selected = rng.choice(choices, n, replace=False)
        current = [i in selected for i in range(size)]
        if predictor.can_predict(current):
            return True, used + 1, tuple(current)
    return False, max_samples, tuple()


def __sample__(
    n: int,
    size: int,
    predictor: Predictor,
    seed: Union[int, np.random.Generator],
    max_samples: int,
) -> Tuple[bool, int, Tuple[bool, ...]]:
    choices = list(range(size))
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    for used in range(max_samples):
        selected = rng.choice(choices, n, replace=False)
        current = [i in selected for i in range(size)]
        if predictor.can_predict(current):
            return True, used + 1, tuple(current)
    return False, max_samples, tuple()


class RandomSamplingSolver(Solver):
    """
    A solver that samples solutions naively.

    """

    def __init__(self, try_improve: bool = False) -> None:
        self.try_improve = try_improve

    def get_name(self) -> str:
        if self.try_improve:
            return "rs+sbs"
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
        improvement_queue = []

        pbar = ProgressBar(total=samples, name=self.get_name(), use_tqdm=use_tqdm)
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            # Find best among possible children
            queued = seed or 0
            while budget > 0 and best_cost > 1:
                while len(futures) < nprocs:
                    if improvement_queue:
                        futures.append(
                            pool.submit(
                                __improve_upon__,
                                improvement_queue.pop(),
                                queued,
                                predictor,
                            )
                        )
                    else:
                        futures.append(
                            pool.submit(
                                __sample__,
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
                        improvement_queue,
                        pbar,
                        self.try_improve,
                    )
            pool.shutdown()

        else:
            while budget > 0 and best_cost > 1:
                if improvement_queue:
                    has_found, used, out = __improve_upon__(
                        improvement_queue.pop(),
                        (seed or 0) + budget,
                        predictor,
                        min(SAMPLING_UNIT, budget),
                    )
                else:
                    has_found, used, out = __sample__(
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
                    improvement_queue,
                    pbar,
                    self.try_improve,
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
