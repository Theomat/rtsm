from typing import Callable, Optional, Set, Tuple, Union

import numpy as np

from rtsm.instance import Instance
from rtsm.predictors.predictor import Predictor, ranking_error, to_ranking
from rtsm.solution import Solution
from rtsm.solvers.solver import Solver
from rtsm.utils.color_helper import get_color_helper

import pygad

F = get_color_helper()


def __sol_conv__(mapping, sol: np.ndarray) -> Tuple[bool, ...]:
    return tuple(mapping[i] if val else 0 for i, val in enumerate(sol))


def __new_sol__(
    sol: Tuple[bool, ...],
    current_best: int,
    solutions: Set[Tuple[bool, ...]],
    on_progress_callback: Optional[Callable[[Set[Solution]], None]] = None,
):
    score = sum(sol)
    if score < current_best:
        solutions.clear()
        solutions.add(sol)
        if on_progress_callback is not None:
            on_progress_callback(solutions)
        return score
    elif score == current_best and sol not in solutions:
        solutions.add(sol)
    return current_best


def __fitness_function__(instance: pygad.GA, sol: np.ndarray, sol_index: int) -> float:
    target_ranking = instance.target_ranking
    predictor = instance.predictor
    (n, acc) = instance.args
    error = ranking_error(
        target_ranking, predictor.get_ranking(__sol_conv__(instance.mapping, sol))
    )
    kept = np.sum(sol) / n
    if error > 1 - acc:
        return 1 - error
    return 1 - kept + 10


def __generation_callback__(instance: pygad.GA):
    current_sol = __sol_conv__(instance.mapping, instance.best_solution()[0])
    instance.best_cost[0] = __new_sol__(
        current_sol,
        instance.best_cost[0],
        instance.best_sol,
        instance.progress_callback,
    )
    # instance.plot_fitness()
    # instance.plot_new_solution_rate()
    if instance.verbose:
        cost = instance.best_cost[0]
        print(
            f"{instance.prefix}{F.LIGHTCYAN_EX}[gen. {instance.generations_completed}]{F.RESET} current best: {F.LIGHTCYAN_EX}{cost}{F.RESET} ({F.LIGHTCYAN_EX}{cost / instance.args[0]:.1%}{F.RESET})"
        )


class GASolver(Solver):
    """
    A solver that uses Genetic Algorithms .

    """

    def get_name(self) -> str:
        return "ga"

    def solve(
        self,
        instance: Instance,
        predictor_builder: Union[Callable[[Instance], Predictor], Predictor],
        use_tqdm: bool = False,
        nprocs: int = 1,
        samples: int = 10000,
        seed: Optional[int] = None,
        verbose: bool = False,
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
        n = len(instance.tests)
        init = instance.warm_start()
        self.best_sol = {init}
        best_cost = [sum(init)]
        pop_size = min(best_cost[0], 10)
        if verbose:
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} init: {F.LIGHTCYAN_EX}{best_cost[0]}{F.RESET} ({F.LIGHTCYAN_EX}{best_cost[0] / len(init):.1%}{F.RESET})"
            )
            print(
                f"{self._get_print_prefix_()}{F.LIGHTCYAN_EX}[info]{F.RESET} pop. size: {F.LIGHTCYAN_EX}{pop_size}{F.RESET}"
            )
        mapping = {}
        j = 0
        for i, val in enumerate(init):
            if val:
                mapping[j] = i
                j += 1

        target_ranking = to_ranking(np.sum(instance.performance_matrix, axis=-1))
        acc = getattr(predictor, "accuracy", 1)

        ga_instance = pygad.GA(
            num_generations=samples,
            num_parents_mating=2,
            fitness_func=__fitness_function__,
            sol_per_pop=pop_size,
            num_genes=best_cost[0],
            gene_space=[0, 1],
            on_generation=__generation_callback__,
            mutation_by_replacement=True,
            parallel_processing=None if nprocs <= 1 else ["process", nprocs],
            random_seed=seed,
            save_solutions=False,
        )
        self.ga_instance = ga_instance
        ga_instance.progress_callback = on_progress_callback
        ga_instance.predictor = predictor
        ga_instance.target_ranking = target_ranking
        ga_instance.args = (n, acc)
        ga_instance.mapping = mapping
        ga_instance.best_cost = best_cost
        ga_instance.best_sol = self.best_sol
        ga_instance.verbose = verbose
        ga_instance.prefix = self._get_print_prefix_()
        ga_instance.run()

        out = self.__get_solutions__()
        del self.ga_instance
        return out

    def __get_solutions__(self) -> Set[Solution]:
        if not hasattr(self, "ga_instance"):
            return set()
        sol, __, ___ = self.ga_instance.best_solution()
        return {
            Solution(
                self.instance,
                tuple(
                    self.instance.get_tests(__sol_conv__(self.ga_instance.mapping, sol))
                ),
            )
        }

    def early_exit(self) -> Set[Solution]:
        return self.__get_solutions__()
