from abc import ABC, abstractmethod
from typing import Tuple

from rtsm.instance import Instance


class Predictor(ABC):
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this solver.
        """
        pass

    @abstractmethod
    def can_predict(self, usable: Tuple[bool, ...]) -> bool:
        """
        Return true if and only if the performances are predictable according to this predictor.
        """
        pass
