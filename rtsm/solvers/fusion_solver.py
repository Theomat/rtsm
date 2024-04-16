from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, Callable, Set, Tuple

import tqdm
from colorama import Fore as F

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solvers.solver import Solver
from rtsm.solution import Solution


def __solve__(
    id: int,
    sub_solver: Solver,
    sub_instance: Instance,
    predictor: Predictor,
    use_tqdm: bool,
    **kwargs,
) -> Tuple[int, Set[Solution]]:
    return id, sub_solver.solve(sub_instance, predictor, use_tqdm, **kwargs)


class SplitManager:
    def __init__(self, instance: Instance, splits: int) -> None:
        self.instance = instance
        instances = instance.split(splits, seed=1)
        self.solutions = {i: v.tests for i, v in enumerate(instances)}
        self.queue = [(k, v) for k, v in enumerate(instances)]
        self.dependencies = {}
        self.merge_queue = []
        self.id_generator = len(self.solutions)

    def has_next(self) -> bool:
        return len(self.queue) > 0

    def is_done(self) -> bool:
        return len(self.solutions) == 1

    def next_instance(self) -> Tuple[int, Instance]:
        return self.queue.pop(0)

    def feed(self, data: Tuple[int, Set[Solution]]) -> None:
        id, sols = data
        # Update solution
        solution = list(sols)[0].tests
        self.solutions[id] = solution
        if id in self.dependencies:
            for x in self.dependencies[id]:
                del self.solutions[x]

        self.merge_queue.append(id)
        # Update merge queue
        if len(self.merge_queue) >= 2:
            a, b = self.merge_queue.pop(), self.merge_queue.pop()

            new_id = self.id_generator
            self.id_generator += 1

            self.dependencies[new_id] = [a, b]

            self.queue.append(
                (new_id, self.instance.subset(self.solutions[a] + self.solutions[b]))
            )

    def current_best_score(self) -> int:
        return sum(len(sol) for sol in self.solutions.values())

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
        verbose: bool = False,
        sub_verbose: bool = False,
        **kwargs: Any,
    ) -> Set[Solution]:
        self.instance = instance
        n = len(instance.tests)
        if use_tqdm:
            pbar = tqdm.tqdm(total=self.splits * 2 - 1, smoothing=0, desc="fusion")
        self.split_manager = SplitManager(instance, self.splits)
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            while not self.split_manager.is_done():
                while len(futures) < nprocs and self.split_manager.has_next():
                    id, sub_instance = self.split_manager.next_instance()
                    futures.append(
                        pool.submit(
                            __solve__,
                            id,
                            self.solver_builder(),
                            sub_instance,
                            predictor_builder(sub_instance),
                            False,
                            verbose=sub_verbose,
                            **kwargs,
                        )
                    )
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    self.split_manager.feed(future.result())
                    futures.remove(future)
                    if use_tqdm:
                        pbar.update(1)
                        score = self.split_manager.current_best_score()
                        pbar.set_postfix_str(
                            f"best: {F.LIGHTYELLOW_EX}{score}{F.RESET} ({F.LIGHTYELLOW_EX}{score/n:.1%}{F.RESET})"
                        )
            pool.shutdown()
        else:
            sub_solver = self.solver_builder()
            while self.split_manager.has_next():
                id, sub_instance = self.split_manager.next_instance()
                out = sub_solver.solve(
                    sub_instance,
                    predictor_builder,
                    use_tqdm,
                    verbose=sub_verbose,
                    **kwargs,
                )
                self.split_manager.feed((id, out))
                if use_tqdm:
                    pbar.update(1)
                    score = self.split_manager.current_best_score()
                    pbar.set_postfix_str(
                        f"best: {F.LIGHTYELLOW_EX}{score}{F.RESET} ({F.LIGHTYELLOW_EX}{score/n:.1%}{F.RESET})"
                    )
        if use_tqdm:
            pbar.close()
        return self.__get_solutions__()

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return self.split_manager.get_solutions()
