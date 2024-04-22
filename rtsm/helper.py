from typing import Callable, Dict, Set, Tuple, List

import numpy as np

from rtsm.data_loaders.csv_data_loader import CSVDataLoader
from rtsm.data_loaders.data_loader import DataLoader
from rtsm.instance import Instance

from rtsm.predictors.predictor import Predictor
from rtsm.predictors.logistic_boolean_predictor import LogisticBooleanPredictor
from rtsm.predictors.logistic_rank_predictor import LogisticRankPredictor
from rtsm.predictors.linear_predictor import LinearRegressionPredictor
from rtsm.solvers.bisect_solver import BisectSolver
from rtsm.solvers.rs_solver import RandomSamplingSolver
from rtsm.solvers.solver import Solver


def get_data_loaders() -> Tuple[List[DataLoader], Set[str]]:
    data_loaders = [CSVDataLoader()]
    supported_extensions = set()
    for loader in data_loaders:
        supported_extensions.update(loader.get_extensions())
    return data_loaders, supported_extensions


def get_predictors() -> Dict[str, Callable[[Instance], Predictor]]:
    def adaptative_predictor(inst: Instance, **kwargs) -> Predictor:
        if np.unique(inst.performance_matrix).shape[0] == 2:
            return LogisticBooleanPredictor(inst, **kwargs)
        return LinearRegressionPredictor(inst, **kwargs)

    return {
        "auto": adaptative_predictor,
        "logistic-bool": LogisticBooleanPredictor,
        "logistic-rank": LogisticRankPredictor,
        "linear": LinearRegressionPredictor,
        "linear+": lambda x, **kwargs: LinearRegressionPredictor(
            x, positive=True, **kwargs
        ),
    }


def get_solvers() -> Dict[str, Solver]:
    __solver_list__: List[Solver] = [BisectSolver(), RandomSamplingSolver()]
    return {solver.get_name(): solver for solver in __solver_list__}
