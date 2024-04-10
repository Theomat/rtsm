from typing import List
import csv

from rtsm.instance import Instance
from rtsm.data_loaders.data_loader import DataLoader


class CSVDataLoader(DataLoader):
    def get_extensions(self) -> List[str]:
        return [".csv"]

    def load(self, path: str) -> Instance:
        with open(path) as fd:
            rows = [row for row in csv.reader(fd)]
        names = rows.pop(0)
        indices = [
            names.index("variant"),
            names.index("test"),
            names.index("performance"),
        ]
        # Find the sets
        variants = set()
        tests = set()
        for row in rows:
            tests.add(row[indices[1]])
            variants.add(row[indices[0]])
        instance = Instance(list(variants), list(tests))
        for row in rows:
            instance.store_performance(
                row[indices[0]], row[indices[1]], float(row[indices[2]])
            )
        return instance
