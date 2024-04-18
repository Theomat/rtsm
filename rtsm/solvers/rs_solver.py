from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, Optional, Set, Tuple, Union

import numpy as np
import tqdm
from colorama import Fore as F

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver


def __new_sol__(
    sol: Tuple[bool, ...],
    current_best: int,
    solutions: Set[Tuple[bool, ...]],
    pbar: Optional[tqdm.tqdm],
):
    score = sum(sol)
    if score < current_best:
        if pbar is not None:
            pbar.set_postfix_str(
                f"best: {F.LIGHTYELLOW_EX}{score}{F.RESET} ({F.LIGHTYELLOW_EX}{score/len(sol):.1%}{F.RESET})"
            )
        solutions.clear()
        solutions.add(sol)
        return score
    elif score == current_best and sol not in solutions:
        solutions.add(sol)
    return current_best


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

        budget = samples

        if use_tqdm:
            pbar = tqdm.tqdm(total=samples, smoothing=0)
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            # Find best among possible children
            queued = seed
            while budget > 0 and best_cost > 1:
                while len(futures) < nprocs:
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
                    if use_tqdm:
                        pbar.update(used)
                    if not has_found:
                        continue
                    best_cost = __new_sol__(
                        out,
                        best_cost,
                        self.best_sol,
                        pbar if use_tqdm else None,
                    )
            pool.shutdown()

        else:
            while budget > 0 and best_cost > 1:
                has_found, used, out = __sample__(
                    best_cost - 1,
                    n,
                    predictor,
                    seed + budget,
                    min(SAMPLING_UNIT, budget),
                )
                budget -= used
                if use_tqdm:
                    pbar.update(used)
                if not has_found:
                    continue
                best_cost = __new_sol__(
                    out,
                    best_cost,
                    self.best_sol,
                    pbar if use_tqdm else None,
                )
        if use_tqdm:
            pbar.close()

        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return {
            Solution(self.instance, tuple(self.instance.get_tests(sol)))
            for sol in self.best_sol
        }

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()
