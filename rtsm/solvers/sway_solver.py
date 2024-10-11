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
        solutions.append(sol)
        if on_progress_callback is not None:
            on_progress_callback([converter(sol)])
        return score
    elif score == current_best:
        solutions.append(sol)
    return current_best


def __sample_candidates__(
    n: int,
    size: int,
    predictor: Predictor,
    seed: Union[int, np.random.Generator],
    max_samples: int,
) -> Generator[np.ndarray, None, None]:
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    choices = list(range(size))
    for used in range(max_samples):
        size_of_sample = rng.integers(1, size)
        rng.shuffle(choices)
        selected = choices[:size_of_sample]
        current = np.asarray([i in selected for i in range(size)])
        yield current


def __is_better__(
    a: List[np.ndarray], b: List[np.ndarray], predictor: Predictor, safe
) -> Tuple[Optional[bool], List[np.ndarray], List[np.ndarray]]:
    """Return a > b?"""
    fa = [x for x in a if safe or predictor.can_predict(x)]
    fb = [x for x in b if safe or predictor.can_predict(x)]
    if len(fa) == len(fb):
        return None, fa, fb
    else:
        return len(fa) > len(fb), fa, fb


def __sway__(
    candidates: List[np.ndarray],
    seed: np.random.Generator,
    predictor: Predictor,
    safe: bool = False,
    target_number: int = 100,
):
    if len(candidates) <= target_number:
        return candidates
    else:
        delta_1 = []
        delta_2 = []
        west, east = __split__(candidates, seed)
        west_better, west, east = __is_better__(west, east, predictor, safe)
        if west_better is None:
            delta_1 = __sway__(west, seed, predictor, True, target_number)
            delta_2 = __sway__(east, seed, predictor, True, target_number)
        elif west_better:
            delta_1 = __sway__(west, seed, predictor, True, target_number)
        else:
            delta_2 = __sway__(east, seed, predictor, True, target_number)
        return delta_1 + delta_2


def __distance__(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(a != b)


def __split__(
    candidates: List[np.ndarray],
    seed: Union[int, np.random.Generator],
    totalGroup: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rand = candidates.pop()
    items = [(x, np.sum(x), __distance__(rand, x)) for x in candidates]
    all_radius = [t[1] for t in items]
    stratified = {}
    for el, r, d in items:
        if r not in stratified:
            stratified[r] = []
        stratified[r].append((d, el))
    for r, elems in stratified.items():
        elems.sort(key=lambda x: x[0])
    thickness = max(stratified.keys()) / totalGroup
    east = []
    west = []
    for a in range(totalGroup):
        elems = [
            i for i in items if (a - 1) * thickness <= i[1] and a * thickness >= i[1]
        ]

        def index(d, el) -> int:
            i = 0
            for dd, ell in stratified[r]:
                if dd == d and np.all(ell == el):
                    return i
                i += 1
            return -1

        elems_with_theta = sorted(
            [(index(d, el) / len(stratified[r]), el) for el, r, d in elems],
            key=lambda x: x[0],
        )
        if elems_with_theta:
            east.append(elems_with_theta.pop(0)[1])
            west.append(elems_with_theta.pop(-1)[1])
            for theta, el in elems_with_theta:
                if theta <= 0.5:
                    east.append(el)
                else:
                    west.append(el)
    return west, east


class SWAYSolver(Solver):
    """
    A solver that use SWAY to find solutions.

    """

    def get_name(self) -> str:
        return "sway"

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

        budget = samples
        units = 20

        def convert(array):
            return Solution(self.instance, tuple(self.instance.get_tests(array)))

        rng = np.random.default_rng(seed)
        SAMPLING_UNIT = samples // units
        target_number = int(np.sqrt(SAMPLING_UNIT))
        pbar = ProgressBar(
            total=units * target_number, name=self.get_name(), use_tqdm=use_tqdm
        )
        pbar.set_best(best_cost, best_cost / len(init))
        for i in range(units):
            sampled = __sample_candidates__(n, best_cost, predictor, rng, SAMPLING_UNIT)
            candidates = __sway__(
                list(sampled), rng, predictor, target_number=target_number
            )
            pbar.update(target_number - len(candidates))

            for candidate in candidates:
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
