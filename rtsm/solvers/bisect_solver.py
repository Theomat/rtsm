from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import tqdm
from colorama import Fore as F


from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver


def __split__(
    current: Tuple[bool, ...],
    must_keep: Tuple[bool, ...],
    n: int,
    rng: np.random.Generator,
) -> Generator[Tuple[bool, ...], None, None]:
    """
    Split current choices in n different partitions.
    I.E. we set some elements of current to 0.
    that is outputs are included in current
    and must_keep are included in outputs
    """
    indices = [i for i, x in enumerate(current) if x and not must_keep[i]]
    rng.shuffle(indices)
    size = len(indices) // n
    parts = n
    while parts > 0:
        if parts == 1:
            yield tuple(must_keep[i] or i in indices for i in range(len(current)))
        else:
            selected = indices[:size]
            indices = indices[size:]
            yield tuple(must_keep[i] or i in selected for i in range(len(current)))
        parts -= 1


def __find_necessary__(
    tcurrent: Tuple[bool, ...], tmust_keep: Tuple[bool, ...], predictor: Predictor
) -> Tuple[Tuple[bool, ...], bool]:
    """
    Use property of stability by inclusion to find out what elements must be kept.
    That is for each element that may benecessary remove it and try to find if it is SAT.
    if it SAT no problem, if it is NOT SAT then it must be necessary.
    """
    current = list(tcurrent)
    must_keep = list(tmust_keep)
    n = len(current)
    fixed = 0
    # Subset test part
    for i in range(n):
        if must_keep[i] or not current[i]:
            fixed += 1
            continue
        current[i] = False
        sat = predictor.can_predict(current)
        must_keep[i] = not sat
        fixed += must_keep[i]
        current[i] = True
    return tuple(must_keep), fixed == n


def __bisect__(
    current: Tuple[bool, ...],
    must_keep: Tuple[bool, ...],
    best_so_far: int,
    seed: Union[int, np.random.Generator],
    predictor: Predictor,
):
    """
    Assume current is SAT
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    while True:
        min_score = sum(must_keep)
        # IF cannot remove element from current OR no better case than best so far
        if min_score >= sum(current) or min_score >= best_so_far:
            return current
        # Split current in 2
        li = list(__split__(current, must_keep, 2, rng))
        a, b = li[0], li[1]
        # If one split is SAT, then great change current to it and update must_keep
        if predictor.can_predict(a):
            current = a
            if min_score <= 1 and sum(a) == 1:
                return a
            must_keep, has_finished = __find_necessary__(current, must_keep, predictor)
            if has_finished:
                return must_keep
        elif predictor.can_predict(b):
            current = b
            if min_score <= 1 and sum(b) == 1:
                return b
            must_keep, has_finished = __find_necessary__(current, must_keep, predictor)
            if has_finished:
                return must_keep
        # Otherwise
        else:
            # then either we must keep all of a or all of b
            # So try to solve either problem and get the best solution
            sol_a = __bisect__(current, a, best_so_far, rng, predictor)
            score_a = sum(sol_a)
            best_so_far = min(score_a, best_so_far)
            sol_b = __bisect__(current, b, best_so_far, rng, predictor)
            score_b = sum(sol_b)
            if score_a < score_b:
                sol = sol_a
            else:
                sol = sol_b
            return sol


def __improve_upon__(
    sol_set: Set[
        Tuple[bool, ...],
    ],
    seed: Union[int, np.random.Generator],
    predictor: Predictor,
) -> Tuple[bool, ...]:
    one_sol = list(sol_set)[0]
    current = tuple(any(sol[i] for sol in sol_set) for i in range(len(one_sol)))
    must_keep = tuple(all(sol[i] for sol in sol_set) for i in range(len(one_sol)))
    return __bisect__(current, must_keep, sum(one_sol), seed, predictor)


def __new_sol__(
    sol: Tuple[bool, ...],
    current_best: int,
    solutions: Set[Tuple[bool, ...]],
    improvement_queue: List[Set[Tuple[bool, ...]]],
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
        for x in solutions:
            improvement_queue.append({x, sol})
        solutions.add(sol)
    return current_best


class BisectSolver(Solver):
    """
    A solver that samples solutions with the help of a bisection algorithm.

    """

    def get_name(self) -> str:
        return "bisect"

    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        verbose: bool = False,
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
        n = len(instance.tests)
        init = instance.warm_start()
        initial_best = sum(init)
        self.best_sol = {init}
        if verbose:
            print(
                f"{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{initial_best}{F.RESET} ({F.LIGHTCYAN_EX}{initial_best / len(init):.1%}{F.RESET})"
            )
        must_keep = __find_necessary__(init, tuple(False for _ in range(n)), predictor)[
            0
        ]
        n_kept = sum(must_keep)
        unfixed = len(must_keep) - n_kept - (len(init) - sum(init))
        if verbose:
            print(
                f"{F.LIGHTCYAN_EX}[info]{F.RESET} top level:\n\tbest possible solution: {F.LIGHTCYAN_EX}{n_kept}{F.RESET} ({F.LIGHTCYAN_EX}{n_kept / initial_best:.1%}{F.RESET})\n\tnot fixed: {F.LIGHTCYAN_EX}{unfixed}{F.RESET} ({F.LIGHTCYAN_EX}{unfixed / initial_best:.1%}{F.RESET})"
            )
        best_possible = max(n_kept, 1)

        improvement_queue = []
        if use_tqdm:
            pbar = tqdm.tqdm(total=samples, smoothing=0)
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            # Find best among possible children
            total_done = 0
            queued = seed or 0
            while total_done < samples and initial_best > best_possible:
                while len(futures) < nprocs and queued < samples:
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
                                __bisect__,
                                tuple(init),
                                must_keep,
                                initial_best,
                                queued,
                                predictor,
                            )
                        )
                    queued += 1
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    total_done += 1
                    out = future.result()
                    futures.remove(future)
                    initial_best = __new_sol__(
                        out,
                        initial_best,
                        self.best_sol,
                        improvement_queue,
                        pbar if use_tqdm else None,
                    )
                    if use_tqdm:
                        pbar.update(1)
            pool.shutdown()
        else:
            for i in range(samples):
                if improvement_queue:
                    out = __improve_upon__(
                        improvement_queue.pop(),
                        i + (seed or 0),
                        predictor,
                    )
                else:
                    out = __bisect__(
                        tuple(init), must_keep, initial_best, i + (seed or 0), predictor
                    )
                initial_best = __new_sol__(
                    out,
                    initial_best,
                    self.best_sol,
                    improvement_queue,
                    pbar if use_tqdm else None,
                )
                if use_tqdm:
                    pbar.update(1)
                if initial_best <= best_possible:
                    break
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
