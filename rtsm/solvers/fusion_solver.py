from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, Callable, List, Optional, Set, Tuple

from colorama import Fore as F
import random

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solvers.solver import Solver
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


class SplitManager:
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
        instances = instance.split(splits, seed=seed)
        self.rng = random.Random(seed)
        self.solutions = {
            i: v.get_tests(v.warm_start()) for i, v in enumerate(instances)
        }
        self.partitions = {i: v.tests[:] for i, v in enumerate(instances)}
        self.queue = [(k, v) for k, v in enumerate(instances)]
        self.tries = {i: 0 for i in range(len(instances))}
        self.max_tries = max_tries
        self.dependencies = {i: [i] for i in range(len(instances))}
        self.merge_queue = []
        self.id_generator = len(self.solutions)

    def has_next(self) -> bool:
        return len(self.queue) > 0

    def smallest_partition(self) -> int:
        return min(len(p) for p in self.partitions.values())

    def is_done(self) -> bool:
        if len(self.queue) == 0:
            self.__update_merge_queue__()
            if len(self.queue) > 0:
                return False
            if len(self.merge_queue) <= 1:
                if len(self.merge_queue) == 1:
                    t = sorted(self.tries.values())
                    return len(t) <= 1 or (t[1] >= self.max_tries)
                else:
                    return True
            else:
                return False
        return False

    def next_instance(self) -> Tuple[int, Instance]:
        return self.queue.pop(0)

    def __accept_one__(
        self, sols: Set[Solution], old_sols: Set[int]
    ) -> Tuple[bool, List[str]]:
        if len(list(sols)[0].tests) == sum(len(self.solutions[x]) for x in old_sols):
            return True, list(sols)[0].tests
        new_partition = []
        for x in old_sols:
            new_partition += self.partitions[x]
        new_inst = self.instance.subset(new_partition)
        predictor = self.predictor_builder(new_inst)
        for sol in sols:
            solution = sol.tests
            # Now we need to check that it works
            if predictor.can_predict([t in solution for t in new_inst.tests]):
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
                (new_id, self.instance.subset(self.solutions[a] + self.solutions[b]))
            )

    def feed(self, data: Tuple[int, Set[Solution]]) -> bool:
        id, sols = data
        # Update solution
        solution = []
        if id in self.dependencies:
            found, solution = self.__accept_one__(sols, self.dependencies[id])
            if not found:
                for x in self.dependencies[id]:
                    self.tries[x] += 1
                    if self.tries[x] < self.max_tries:
                        self.merge_queue.append(x)
                del self.dependencies[id]
                return False
        else:
            solution = list(sols)[0].tests

        # Update dict
        new_partition = []
        for x in self.dependencies[id]:
            del self.solutions[x]
            new_partition += self.partitions[x]
            del self.partitions[x]
            del self.tries[x]
        del self.dependencies[id]
        self.partitions[id] = new_partition
        self.solutions[id] = solution
        self.tries[id] = 0

        self.merge_queue.append(id)
        return True

    def current_best_score(self) -> int:
        return sum(len(sol) for sol in self.solutions.values())

    def get_solutions(self) -> Set[Solution]:
        out = []
        for part in self.solutions.values():
            out += part
        return {Solution(self.instance, tuple(out))}


class SplitChooser:
    def __init__(
        self,
        instance: Instance,
        splits: int,
        seed: Optional[int],
        predictor_builder: Callable[[Instance], Predictor],
        max_tries: int,
    ):
        self.instance = instance
        self.max_tries = max_tries
        self.managers = [
            SplitManager(
                instance,
                splits,
                (i * 17 + (seed or 0)) * 17 + 31,
                predictor_builder,
                max_tries,
            )
            for i in range(10)
        ]
        self.second_phase = False
        self.chosen_one = 0
        self.queue = []
        for i, m in enumerate(self.managers):
            self.queue.append((i, m.next_instance()))

    def current_best_score(self) -> int:
        if self.second_phase:
            return self.managers[self.chosen_one].current_best_score()
        return min(m.current_best_score() for m in self.managers)

    def get_solutions(self) -> Set[Solution]:
        if self.second_phase:
            return self.managers[self.chosen_one].get_solutions()
        out = set()
        for m in self.managers:
            out |= m.get_solutions()
        return out

    def has_next(self) -> bool:
        if self.second_phase:
            return self.managers[self.chosen_one].has_next()
        return len(self.queue) > 0

    def is_done(self) -> bool:
        if self.second_phase:
            return self.managers[self.chosen_one].is_done()
        if len(self.queue) == 0:
            alive = 0
            for i, m in enumerate(self.managers):
                if m.smallest_partition() >= 100:
                    continue
                alive += 1
                if not m.is_done() and m.has_next():
                    self.queue.append((i, m.next_instance()))

            if alive == 0:
                self.second_phase = True
                self.chosen_one = 0
                score = 1e99
                for i, m in enumerate(self.managers):
                    if m.current_best_score() < score:
                        self.chosen_one = i
                        score = m.current_best_score()
                # Keep relevant tasks
                self.queue = [x for x in self.queue if x[0] == self.chosen_one]
                return self.is_done()
            return len(self.queue) == 0 
        return False

    def next_instance(self) -> Tuple[int, Tuple[int, Instance]]:
        if self.second_phase:
            if self.queue:
                return self.queue.pop()
            return self.chosen_one, self.managers[self.chosen_one].next_instance()
        return self.queue.pop()

    def feed(self, data: Tuple[int, int, Set[Solution]]) -> bool:
        manager = self.managers[data[0]]
        manager.feed(data[1:])


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
        **kwargs: Any,
    ) -> Set[Solution]:
        self.instance = instance
        n = len(instance.tests)
        self.split_manager = SplitChooser(
            instance, self.splits, seed, predictor_builder, samples
        )
        if verbose:
            best_score = self.split_manager.current_best_score()
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_score}{F.RESET} ({F.LIGHTCYAN_EX}{best_score/ len(instance.tests):.1%}{F.RESET})"
            )
        pbar = ProgressBar(
            total=(self.splits * 2 - 1) * self.split_manager.max_tries * 10,
            name=self.get_name(),
            use_tqdm=use_tqdm,
        )
        kwargs["seed"] = seed
        kwargs["use_tqdm"] = False
        kwargs["verbose"] = False
        kwargs["samples"] = samples
        left_over = 0
        if nprocs > 1:
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            while not self.split_manager.is_done():
                while len(futures) < nprocs and self.split_manager.has_next():
                    did, (id, sub_instance) = self.split_manager.next_instance()
                    futures.append(
                        pool.submit(
                            __solve__,
                            (did, id),
                            self.solver_builder(),
                            sub_instance,
                            predictor_builder(sub_instance),
                            **kwargs,
                        )
                    )
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    (did, id), out = future.result()
                    accepted = self.split_manager.feed((did, id, out))
                    futures.remove(future)
                    if accepted:
                        todo = max(0, self.split_manager.max_tries - left_over)
                        left_over -= self.split_manager.max_tries - todo
                        pbar.update(todo)
                    else:
                        pbar.update(1)
                        left_over += 1
                score = self.split_manager.current_best_score()
                pbar.set_best(score, score / n)
            pool.shutdown()
        else:
            sub_solver = self.solver_builder()
            while not self.split_manager.is_done():
                did, (id, sub_instance) = self.split_manager.next_instance()
                out = sub_solver.solve(
                    sub_instance,
                    predictor_builder,
                    **kwargs,
                )
                accepted = self.split_manager.feed((did, id, out))
                if accepted:
                    todo = max(0, self.split_manager.max_tries - left_over)
                    left_over -= self.split_manager.max_tries - todo
                    pbar.update(todo)
                else:
                    pbar.update(1)
                    left_over += 1
                score = self.split_manager.current_best_score()
                pbar.set_best(score, score / n)
        pbar.close()
        return self.__get_solutions__()

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return self.split_manager.get_solutions()
