from abc import ABC, abstractmethod
from typing import List

from rtsm.instance import Instance


class DataLoader(ABC):
    @abstractmethod
    def get_extensions(self) -> List[str]:
        """
        List of file extensions supported by this data loader.
        """
        pass

    def match(self, path: str) -> bool:
        """
        Check if the specified file is in a supported format by this loader.
        """
        return any(path.endswith(ext) for ext in self.get_extensions())

    @abstractmethod
    def load(self, path: str) -> Instance:
        """
        Loads the specified file into a RTSM Instance.
        An important requirement is that they are non stochastic, that is the order of the tests and variants in the Instance is deterministic.
        """
        pass
