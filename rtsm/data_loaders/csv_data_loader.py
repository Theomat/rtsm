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
        performances = [name for name in names if name not in ["variant", "test"]]
        indices = [
            names.index("variant"),
            names.index("test"),
        ]
        for perf in performances:
            indices.append(names.index(perf))
        # Find the sets
        variants = set()
        tests = set()
        for row in rows:
            tests.add(row[indices[1]])
            variants.add(row[indices[0]])
        instance = Instance(sorted(performances), sorted(variants), sorted(tests))
        for row in rows:
            for perf, index in zip(performances, indices[2:]):
                instance.store_performance(
                    perf, row[indices[0]], row[indices[1]], float(row[index])
                )
        return instance
