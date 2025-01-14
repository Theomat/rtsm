from typing import List
import csv

from rtsm.instance import Instance
from rtsm.data_loaders.data_loader import DataLoader

SEP = "=" * 80 + "\n"


class CSVDataLoader(DataLoader):
    def get_extensions(self) -> List[str]:
        return [".csv"]

    def load(self, path: str) -> Instance:
        with open(path) as fd:
            lines = fd.readlines()
            if SEP in lines:
                separation_index = lines.index(SEP)
                cost_rows = [row for row in csv.reader(lines[:separation_index])]
                rows = [row for row in csv.reader(lines[separation_index + 1 :])]
            else:
                cost_rows = []
                rows = [row for row in csv.reader(lines)]

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
        if cost_rows:
            names = cost_rows.pop(0)
            indices = [
                names.index("test"),
                1 - names.index("test"),
            ]
            for row in cost_rows:
                instance.store_cost(row[indices[0]], row[indices[1]])
        else:
            for test in tests:
                instance.store_cost(test, 1)
        return instance
