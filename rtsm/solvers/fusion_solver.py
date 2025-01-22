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


class Fusion:
    def __init__(
        self,
        instance: Instance,
        splits: int,
        seed: Optional[int],
        predictor_builder: Callable[[Instance], Predictor],
    ) -> None:
        self.predictor_builder = predictor_builder
        self.instance = instance
        self.splits = splits
        self.seed = seed
        self.rng = random.Random(
            seed
        )  # not numpy since we want to use it to shuffle a queue
        instances = self.instance.split(self.splits, seed=(self.seed or 0))
        self.solutions = {
            i: v.get_tests(v.warm_start()) for i, v in enumerate(instances)
        }
        self.partitions = {i: v.tests[:] for i, v in enumerate(instances)}
        self.queue = [(k, v) for k, v in enumerate(instances)]
        self.dependencies = {i: [i] for i in range(len(instances))}
        self.scores = {
            i: get_cost(instances[i], self.solutions[i]) for i in range(len(instances))
        }
        self.merge_queue = []
        self.id_generator = len(self.solutions)

    def has_next(self) -> bool:
        return len(self.queue) > 0

    def is_done(self) -> bool:
        """
        Return done
        """
        if len(self.queue) == 0:
            self.__update_merge_queue__()
            if len(self.queue) > 0:
                return False
            elif len(self.merge_queue) <= 1:
                return len(self.solutions) == 1
            else:
                return False
        return False

    def next_instance(self) -> Tuple[int, Instance]:
        return self.queue.pop(0)

    def __find_valid_solution__(
        self, sols: Set[Solution], old_dependencies: Set[int]
    ) -> List[str]:
        # If no progress was made
        old_cost = sum(self.scores[x] for x in old_dependencies)
        valid_sols = [sol for sol in sols if sol.cost() < old_cost]
        if len(valid_sols) == 0:
            old_sol = []
            for x in old_dependencies:
                old_sol += self.solutions[x]
            return old_sol
        else:
            new_partition = []
            for x in old_dependencies:
                new_partition += self.partitions[x]
            new_inst = self.instance.subset(new_partition)
            predictor = self.predictor_builder(new_inst)
            for sol in valid_sols:
                # Now we need to check that it works
                if predictor.can_predict(
                    np.asarray([t in sol.tests for t in new_inst.tests])
                ):
                    return sol.tests
            old_sol = []
            for x in old_dependencies:
                old_sol += self.solutions[x]
            return old_sol

    def __update_merge_queue__(self) -> None:
        self.rng.shuffle(self.merge_queue)
        while len(self.merge_queue) >= 2:
            a, b = self.merge_queue.pop(), self.merge_queue.pop()

            new_id = self.id_generator
            self.id_generator += 1

            self.dependencies[new_id] = [a, b]
            new_instance = self.instance.subset(
                list(self.partitions[a]) + list(self.partitions[b])
            )
            new_instance.set_start(list(self.solutions[a]) + list(self.solutions[b]))
            self.queue.append((new_id, new_instance))

    def feed(self, data: Tuple[int, Set[Solution]]) -> bool:
        id, sols = data
        # Update solution
        solution = []
        if id in self.dependencies:
            solution = self.__find_valid_solution__(sols, self.dependencies[id])
        else:
            solution = list(sols)[0].tests
            assert False, f"{data} \n\nDEPS:\n\t{self.dependencies}"
        # Update dict
        new_partition = []
        for x in self.dependencies[id]:
            del self.solutions[x]
            del self.scores[x]
            new_partition += self.partitions[x]
            del self.partitions[x]
        del self.dependencies[id]
        self.partitions[id] = new_partition
        self.solutions[id] = solution
        self.scores[id] = get_cost(self.instance, solution)

        self.merge_queue.append(id)
        return True

    def current_best_score(self) -> int:
        return sum(s for s in self.scores.values())

    def get_solutions(self) -> Set[Solution]:
        out = []
        for part in self.solutions.values():
            out += part
        return {Solution(self.instance, tuple(out))}


class FusionSolver(Solver):
    """
    This is a meta solver, that uses divide and conquer on top of another solver.
    """

    def __init__(self, solver_builder: Callable[[], Solver], splits: int) -> None:
        self.solver_builder = solver_builder
        self.splits = splits

    def get_name(self) -> str:
        return f"fusion{self.splits}-{self.solver_builder().get_name()}"

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
        self.split_manager = Fusion(instance, self.splits, seed, predictor_builder)
        best_score = self.split_manager.current_best_score()
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_score}{F.RESET} ({F.LIGHTCYAN_EX}{best_score/ total_cost:.1%}{F.RESET})"
            )
        pbar = ProgressBar(
            total=(self.splits * 2 - 1),
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
            while not self.split_manager.is_done():
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
            pool.shutdown()
        else:
            sub_solver = self.solver_builder()
            while not self.split_manager.is_done():
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
