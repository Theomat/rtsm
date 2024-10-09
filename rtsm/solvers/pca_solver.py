import numpy as np

from rtsm.predictors.predictor import Predictor
from rtsm.solvers.deterministic_solver import DeterministicSolver

from sklearn.decomposition import PCA


class PCASolver(DeterministicSolver):
    """
    A solver that PCA chooses tests, only works for 1 performance.

    """

    def get_name(self) -> str:
        return "pca"

    def __choose_index_to_remove__(
        self, current: np.ndarray, predictor: Predictor
    ) -> int:
        assert (
            self.instance.performance_matrix.shape[0] == 1
        ), "PCASolver: does not support multiple performance measures"
        X = self.instance.performance_matrix[0, :, current].T
        model = PCA()
        Xt = model.fit_transform(X)
        mini_index = np.abs(model.components_[0]).argmin()
        return mini_index
