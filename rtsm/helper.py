from typing import Callable, Dict, Set, Tuple, List
import sys

import numpy as np
from colorama import Fore as F

from rtsm.data_loaders.csv_data_loader import CSVDataLoader
from rtsm.data_loaders.data_loader import DataLoader
from rtsm.instance import Instance

from rtsm.predictors.predictor import Predictor
from rtsm.predictors.no_predictor import NoPredictor
from rtsm.predictors.logistic_boolean_predictor import LogisticBooleanPredictor
from rtsm.predictors.linear_predictor import LinearRegressionPredictor
from rtsm.solvers.bisect_solver import BisectSolver
from rtsm.solvers.greedy_solver import GreedySolver
from rtsm.solvers.pca_solver import PCASolver
from rtsm.solvers.sway_solver import SWAYSolver
from rtsm.solvers.stupid_solver import StupidSolver
from rtsm.solvers.friedman_solver import FriedmanSolver
from rtsm.solvers.rs_solver import RandomSolutionSolver
from rtsm.solvers.solver import Solver


def get_data_loaders() -> Tuple[List[DataLoader], Set[str]]:
    data_loaders = [CSVDataLoader()]
    supported_extensions = set()
    for loader in data_loaders:
        supported_extensions.update(loader.get_extensions())
    return data_loaders, supported_extensions


def try_load_instance(
    path: str, data_loaders: List[DataLoader], swap: bool = False
) -> Instance:
    loaders = [d for d in data_loaders if d.match(path)]
    if len(loaders) == 0:
        print(f"{F.RED}file format not supported:{F.RESET}", path, file=sys.stderr)
        sys.exit(1)
    instance = loaders.pop(0).load(path)
    # Swap instance
    if swap:
        instance = instance.swap()
    if not instance.check_filled():
        print(
            f"{F.YELLOW}warning:{F.RESET} the performance matrix is not completely filled!"
        )
    return instance


def get_predictors() -> Dict[str, Callable[[Instance], Predictor]]:
    def adaptative_predictor(inst: Instance, **kwargs) -> Predictor:
        if np.unique(inst.performance_matrix).shape[0] == 2:
            return LogisticBooleanPredictor(inst, **kwargs)
        return LinearRegressionPredictor(inst, **kwargs)

    return {
        "auto": adaptative_predictor,
        "none": NoPredictor,
        "logistic-bool": LogisticBooleanPredictor,
        "linear": LinearRegressionPredictor,
        "linear+": lambda x, **kwargs: LinearRegressionPredictor(
            x, positive=True, **kwargs
        ),
    }


def get_solvers() -> Dict[str, Solver]:
    __solver_list__: List[Solver] = [
        BisectSolver(False),
        RandomSolutionSolver(),
        BisectSolver(True),
        GreedySolver(),
        PCASolver(),
        FriedmanSolver(),
        SWAYSolver(),
        StupidSolver(),
    ]
    try:
        from rtsm.solvers.ga_solver import GASolver

        __solver_list__.append(GASolver())
    except:
        pass

    return {solver.get_name(): solver for solver in __solver_list__}
