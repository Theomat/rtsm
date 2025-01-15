from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, Callable, List, Optional, Set, Tuple

from colorama import Fore as F
import random

import numpy as np

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solvers.solver import Solver, get_cost
from rtsm.solution import Solution
from rtsm.utils.progress_bar import ProgressBar


def __solve__(
    id: int,
    sub_solver: Solver,
    sub_instance: Instance,
    predictor: Predictor,
    use_tqdm: bool,
    **kwargs,
) -> Tuple[int, Set[Solution]]:
    return id, sub_solver.solve(sub_instance, predictor, use_tqdm, **kwargs)


class SplitRetry:
    def __init__(
        self,
        instance: Instance,
        splits: int,
        seed: Optional[int],
        predictor_builder: Callable[[Instance], Predictor],
        max_tries: int,
    ) -> None:
        self.predictor_builder = predictor_builder
        self.instance = instance
        self.t = max_tries
        self.splits = splits
        self.seed = seed
        self.rng = random.Random(
            seed
        )  # not numpy since we want to use it to shuffle a queue
        self.__new_try__()

    def __new_try__(self):
        self.t -= 1
        instances = self.instance.split(self.splits, seed=(self.seed or 0) + self.t)
        self.solutions = {
            i: v.get_tests(v.warm_start()) for i, v in enumerate(instances)
        }
        self.partitions = {i: v.tests[:] for i, v in enumerate(instances)}
        self.queue = [(k, v) for k, v in enumerate(instances)]
        self.dependencies = {i: [i] for i in range(len(instances))}
        self.merge_queue = []
        self.id_generator = len(self.solutions)

    def has_next(self) -> bool:
        return len(self.queue) > 0

    def smallest_partition(self) -> int:
        return min(len(p) for p in self.partitions.values())

    def is_done(self) -> Tuple[bool, bool]:
        """
        Return (done, has_started_a_new_try)
        """
        if len(self.queue) == 0:
            self.__update_merge_queue__()
            if len(self.queue) > 0:
                return False, False
            if len(self.merge_queue) <= 1:
                if self.t > 0:
                    one_sol = list(self.get_solutions())[0].tests
                    self.instance.set_start(one_sol)
                    self.__new_try__()
                    return False, True
                return True, False
            else:
                return False, False
        return False, False

    def next_instance(self) -> Tuple[int, Instance]:
        return self.queue.pop(0)

    def __accept_one__(
        self, sols: Set[Solution], old_dependencies: Set[int]
    ) -> Tuple[bool, List[str]]:
        # If no progress was made
        new_cost = get_cost(self.instance, list(sols)[0].tests)
        if new_cost >= sum(
            get_cost(self.instance, self.solutions[x]) for x in old_dependencies
        ):
            return True, list(sols)[0].tests
        new_partition = []
        for x in old_dependencies:
            new_partition += self.partitions[x]
        new_inst = self.instance.subset(new_partition)
        predictor = self.predictor_builder(new_inst)
        for sol in sols:
            solution = sol.tests
            # Now we need to check that it works
            if predictor.can_predict(
                np.asarray([t in solution for t in new_inst.tests])
            ):
                return True, solution
        return False, []

    def __update_merge_queue__(self) -> None:
        self.rng.shuffle(self.merge_queue)
        while len(self.merge_queue) >= 2:
            a, b = self.merge_queue.pop(), self.merge_queue.pop()

            new_id = self.id_generator
            self.id_generator += 1

            self.dependencies[new_id] = [a, b]

            self.queue.append(
                (
                    new_id,
                    self.instance.subset(
                        list(self.solutions[a]) + list(self.solutions[b])
                    ),
                )
            )

    def feed(self, data: Tuple[int, Set[Solution]]) -> bool:
        id, sols = data
        # Update solution
        solution = []
        if id in self.dependencies:
            found, solution = self.__accept_one__(sols, self.dependencies[id])
            if not found:
                del self.dependencies[id]
                return False
        else:
            solution = list(sols)[0].tests
            assert False, f"{data} \n\nDEPS:\n\t{self.dependencies}"
        # Update dict
        new_partition = []
        for x in self.dependencies[id]:
            del self.solutions[x]
            new_partition += self.partitions[x]
            del self.partitions[x]
        del self.dependencies[id]
        self.partitions[id] = new_partition
        self.solutions[id] = solution

        self.merge_queue.append(id)
        return True

    def current_best_score(self) -> int:
        return sum(get_cost(self.instance, sol) for sol in self.solutions.values())

    def get_solutions(self) -> Set[Solution]:
        out = []
        for part in self.solutions.values():
            out += part
        return {Solution(self.instance, tuple(out))}


class RetrySolver(Solver):
    """
    This is a meta solver, that uses divide and conquer on top of another solver.
    """

    def __init__(self, solver_builder: Callable[[], Solver], splits: int) -> None:
        self.solver_builder = solver_builder
        self.splits = splits

    def get_name(self) -> str:
        return f"retry{self.splits}-{self.solver_builder().get_name()}"

    def solve(
        self,
        instance: Instance,
        predictor_builder: Callable[[Instance], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        seed: Optional[int] = None,
        samples: int = 20,
        verbose: bool = False,
        on_progress_callback: Optional[Callable[[Set[Solution]], None]] = None,
        **kwargs: Any,
    ) -> Set[Solution]:
        self.instance = instance
        total_cost = instance.total_cost()
        max_tries = 5
        self.split_manager = SplitRetry(
            instance, self.splits, seed, predictor_builder, max_tries
        )
        best_score = self.split_manager.current_best_score()
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_score}{F.RESET} ({F.LIGHTCYAN_EX}{best_score/ total_cost:.1%}{F.RESET})"
            )
        pbar = ProgressBar(
            total=(self.splits * 2 - 1) * max_tries,
            name=self.get_name(),
            use_tqdm=use_tqdm,
        )
        kwargs["seed"] = seed
        kwargs["use_tqdm"] = False
        kwargs["verbose"] = False
        kwargs["samples"] = samples
        score = best_score
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            is_done = self.split_manager.is_done()[0]
            has_restarted = False
            while not is_done:
                if has_restarted:
                    for future in futures:
                        future.cancel()
                    futures.clear()
                while len(futures) < nprocs and self.split_manager.has_next():
                    (id, sub_instance) = self.split_manager.next_instance()
                    futures.append(
                        pool.submit(
                            __solve__,
                            id,
                            self.solver_builder(),
                            sub_instance,
                            predictor_builder(sub_instance),
                            **kwargs,
                        )
                    )
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    id, out = future.result()
                    accepted = self.split_manager.feed((id, out))
                    futures.remove(future)
                    pbar.update(1)
                    if accepted:
                        if (
                            self.split_manager.current_best_score() < score
                            and on_progress_callback is not None
                        ):
                            on_progress_callback(self.split_manager.get_solutions())
                score = self.split_manager.current_best_score()
                pbar.set_best(score, score / total_cost)
                is_done, has_restarted = self.split_manager.is_done()
            pool.shutdown()
        else:
            sub_solver = self.solver_builder()
            while not self.split_manager.is_done()[0]:
                (id, sub_instance) = self.split_manager.next_instance()
                out = sub_solver.solve(
                    sub_instance,
                    predictor_builder,
                    **kwargs,
                )
                accepted = self.split_manager.feed((id, out))
                pbar.update(1)
                if accepted:
                    if (
                        self.split_manager.current_best_score() < score
                        and on_progress_callback is not None
                    ):
                        on_progress_callback(self.split_manager.get_solutions())
                score = self.split_manager.current_best_score()
                pbar.set_best(score, score / total_cost)
        pbar.close()
        return self.__get_solutions__()

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return self.split_manager.get_solutions()
