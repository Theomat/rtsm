from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Optional, Set, Tuple, Union

import numpy as np
import tqdm

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
            pbar.set_postfix_str(f"best: {current_best} ({current_best/len(sol):.1%})")
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
        predictor: Predictor,
        use_tqdm: bool = False,
        nprocs: int = 1,
        samples: int = 10000,
        **kwargs,
    ) -> Set[Solution]:
        """
        Try to solve an instance of RTSM and provides a set of solutions.
        """
        SAMPLING_UNIT = 100
        n = len(instance.tests)
        init = tuple(False for _ in range(n))
        best_sol = {init}
        best_cost = n

        budget = samples

        solutions = defaultdict(int)
        if use_tqdm:
            pbar = tqdm.tqdm(total=samples, smoothing=0)
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            # Find best among possible children
            total_done = 0
            queued = 0
            while total_done < samples:
                while len(futures) < nprocs and budget < samples:
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
                    total_done += 1
                    has_found, used, out = future.result()
                    futures.remove(future)
                    budget -= used
                    if use_tqdm:
                        pbar.update(used)
                    if not has_found:
                        continue
                    solutions[sum(out)] += 1
                    best_cost = __new_sol__(
                        out,
                        best_cost,
                        best_sol,
                        pbar if use_tqdm else None,
                    )
            pool.shutdown()

        else:
            while budget > 0:
                has_found, used, out = __sample__(
                    best_cost - 1, n, predictor, budget, min(SAMPLING_UNIT, budget)
                )
                budget -= used
                if not has_found:
                    continue
                solutions[sum(out)] += 1
                best_cost = __new_sol__(
                    out,
                    best_cost,
                    best_sol,
                    pbar if use_tqdm else None,
                )
        if use_tqdm:
            pbar.close()
        return {Solution(instance, instance.get_tests(sol)) for sol in solutions}
