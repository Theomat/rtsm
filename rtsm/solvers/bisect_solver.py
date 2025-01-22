from concurrent.futures import ProcessPoolExecutor, wait
from typing import Callable, Generator, List, Optional, Set, Union

import numpy as np

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver, get_cost
from rtsm.utils.color_helper import get_color_helper
from rtsm.utils.progress_bar import ProgressBar

F = get_color_helper()


def __split__(
    current: np.ndarray,
    must_keep: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> Generator[np.ndarray, None, None]:
    """
    Split current choices in n different partitions.
    I.E. we set some elements of current to 0.
    that is outputs are included in current
    and must_keep are included in outputs
    """
    indices = np.nonzero(current & (~must_keep))[0]
    rng.shuffle(indices)
    size = len(indices) // n
    parts = n
    while parts > 0:
        if parts > 1:
            selected = indices[size * (n - parts) : size * (n + 1 - parts)]
        else:
            selected = indices[size * (n - 1) :]
        r = must_keep.copy()
        r[selected] = True
        yield r
        parts -= 1


def __find_necessary__(
    current: np.ndarray, must_keep: np.ndarray, predictor: Predictor
) -> bool:
    """
    Use property of stability by inclusion to find out what elements must be kept.
    That is for each element that may be ecessary remove it and try to find if it is SAT.
    if it SAT no problem, if it is NOT SAT then it must be necessary.
    """
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
    return fixed == n


def __bisect__(
    current: np.ndarray,
    must_keep: np.ndarray,
    best_so_far: int,
    seed: Union[int, np.random.Generator],
    predictor: Predictor,
) -> np.ndarray:
    """
    Assume current is SAT
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    instance = predictor.instance
    while True:
        min_score = get_cost(instance, must_keep)
        # IF cannot remove element from current OR no better case than best so far
        if min_score >= get_cost(instance, current) or min_score >= best_so_far:
            return current
        # Split current in 2
        li = list(__split__(current, must_keep, 2, rng))
        a, b = li[0], li[1]
        # If one split is SAT, then great change current to it and update must_keep
        if predictor.can_predict(a):
            current = a
            if min_score <= 1 and np.sum(a) == 1:
                return a
            has_finished = __find_necessary__(current, must_keep, predictor)
            if has_finished:
                return must_keep
        elif predictor.can_predict(b):
            current = b
            if min_score <= 1 and np.sum(b) == 1:
                return b
            has_finished = __find_necessary__(current, must_keep, predictor)
            if has_finished:
                return must_keep
        # Otherwise
        else:
            # then either we must keep all of a or all of b
            # So try to solve either problem and get the best solution
            sol_a = __bisect__(current.copy(), a, best_so_far, rng, predictor)
            score_a = get_cost(instance, sol_a)
            best_so_far = min(score_a, best_so_far)
            sol_b = __bisect__(current, b, best_so_far, rng, predictor)
            score_b = get_cost(instance, sol_b)
            if score_a < score_b:
                sol = sol_a
            else:
                sol = sol_b
            return sol


def __improve_upon__(
    sol_set: List[np.ndarray,],
    seed: Union[int, np.random.Generator],
    predictor: Predictor,
) -> np.ndarray:
    one_sol = list(sol_set)[0]
    current = np.asarray([any(sol[i] for sol in sol_set) for i in range(len(one_sol))])
    must_keep = np.asarray(
        [all(sol[i] for sol in sol_set) for i in range(len(one_sol))]
    )
    return __bisect__(
        current, must_keep, get_cost(predictor.instance, one_sol), seed, predictor
    )


def __new_sol__(
    sol: np.ndarray,
    current_best: int,
    solutions: List[np.ndarray],
    improvement_queue: List[List[np.ndarray]],
    pbar: ProgressBar,
    try_improve: bool,
    converter: Callable[[np.ndarray], Solution],
    instance: Instance,
    on_progress_callback: Optional[Callable[[Set[Solution]], None]] = None,
):
    score = get_cost(instance, sol)
    if score < current_best:
        pbar.set_best(score, score / instance.total_cost())
        solutions.clear()
        solutions.append(sol)
        if on_progress_callback is not None:
            on_progress_callback([converter(sol)])
        return score
    elif score == current_best:
        if try_improve:
            for x in solutions:
                improvement_queue.append([x, sol])
        solutions.append(sol)
    return current_best


class BisectSolver(Solver):
    """
    A solver that samples solutions with the help of a bisection algorithm.

    """

    def __init__(self, try_improve: bool = False) -> None:
        self.try_improve = try_improve

    def get_name(self) -> str:
        if self.try_improve:
            return "bs+sbs"
        return "bs"

    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        verbose: bool = False,
        samples: int = 10000,
        seed: Optional[int] = None,
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
        init = np.asarray(instance.warm_start())
        initial_best = get_cost(instance, init)
        self.best_sol = [init]
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{initial_best}{F.RESET} ({F.LIGHTCYAN_EX}{initial_best / instance.total_cost():.1%}{F.RESET})"
            )
        must_keep = np.copy(init)
        must_keep[:] = False
        __find_necessary__(init, must_keep, predictor)
        n_kept = np.sum(must_keep)
        unfixed = len(must_keep) - n_kept - (len(init) - np.sum(init))
        best_possible = n_kept - unfixed
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} top level:\n\tbest possible solution: {F.LIGHTCYAN_EX}{n_kept}{F.RESET} ({F.LIGHTCYAN_EX}{n_kept / initial_best:.1%}{F.RESET})\n\tnot fixed: {F.LIGHTCYAN_EX}{unfixed}{F.RESET} ({F.LIGHTCYAN_EX}{unfixed / np.sum(init):.1%}{F.RESET})"
            )

        def convert(array):
            return Solution(self.instance, tuple(self.instance.get_tests(array)))

        improvement_queue = []
        pbar = ProgressBar(total=samples, name=self.get_name(), use_tqdm=use_tqdm)
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            # Find best among possible children
            total_done = 0
            queued = seed or 0
            while total_done < samples and np.sum(self.best_sol[0]) > best_possible:
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
                                init.copy(),
                                must_keep.copy(),
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
                        pbar,
                        self.try_improve,
                        convert,
                        instance,
                        on_progress_callback,
                    )
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
                        init.copy(),
                        must_keep.copy(),
                        initial_best,
                        i + (seed or 0),
                        predictor,
                    )
                initial_best = __new_sol__(
                    out,
                    initial_best,
                    self.best_sol,
                    improvement_queue,
                    pbar,
                    self.try_improve,
                    convert,
                    instance,
                    on_progress_callback,
                )
                pbar.update(1)
                if np.sum(self.best_sol[0]) <= best_possible:
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
