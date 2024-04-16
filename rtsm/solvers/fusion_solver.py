from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, Callable, Set

import tqdm
from colorama import Fore as F

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor
from rtsm.solvers.solver import Solver
from rtsm.solution import Solution


def __solve__(
    sub_solver: Solver,
    sub_instance: Instance,
    predictor: Predictor,
    use_tqdm: bool,
    **kwargs,
) -> Set[Solution]:
    return sub_solver.solve(sub_instance, predictor, use_tqdm, **kwargs)


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
        **kwargs: Any,
    ) -> Set[Solution]:
        sub_instances = instance.split(self.splits, seed=1)
        self.instance = instance
        n = len(instance.tests)
        init = tuple(True for _ in range(n))
        self.best_sol = {init}
        if use_tqdm:
            pbar = tqdm.tqdm(total=len(instance.tests), smoothing=0, desc="fusion")
        if nprocs > 1:
            kwargs["verbose"] = False
            pool = ProcessPoolExecutor(nprocs)
            futures = []
            # Find best among possible children
            queued_subsol = []
            while len(futures) > 0 or len(sub_instances) > 0 or len(queued_subsol) >= 2:
                while len(futures) < nprocs and (
                    len(queued_subsol) >= 2 or len(sub_instances) > 0
                ):
                    sub_instance = (
                        sub_instances.pop()
                        if len(queued_subsol) < 2
                        else instance.subset(queued_subsol.pop() + queued_subsol.pop())
                    )
                    futures.append(
                        pool.submit(
                            __solve__,
                            self.solver_builder(),
                            sub_instance,
                            predictor_builder(sub_instance),
                            False,
                            **kwargs,
                        )
                    )
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    out = list(future.result())[0].tests
                    queued_subsol.append(out)
                    futures.remove(future)
                    if use_tqdm:
                        pbar.update(1)
            pool.shutdown()
        else:
            next_sub_instances = []
            sub_solver = self.solver_builder()
            while len(sub_instances) > 1:
                merged_solutions = []
                for sub_instance in sub_instances:
                    solutions = sub_solver.solve(
                        sub_instance,
                        predictor_builder,
                        use_tqdm,
                        **kwargs,
                    )
                    if use_tqdm:
                        pbar.update(1)
                    merged_solutions += list(solutions)[0].tests
                sub_instances = next_sub_instances
                current_instance = instance.subset(merged_solutions)
                sub_instances = current_instance.split(
                    len(sub_instances) // 2, seed=len(sub_instances)
                )
                self.best_sol = {tuple(merged_solutions)}
                if use_tqdm:
                    score = len(merged_solutions)
                    pbar.set_postfix_str(
                        f"best: {F.LIGHTYELLOW_EX}{score}{F.RESET} ({F.LIGHTYELLOW_EX}{score/n:.1%}{F.RESET})"
                    )
        if use_tqdm:
            pbar.close()
        return self.__get_solutions__()

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()

    def __get_solutions__(self) -> Set[Solution]:
        return {Solution(self.instance, sol) for sol in self.best_sol}
