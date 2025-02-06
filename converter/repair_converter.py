import json
import os
import sys
import csv
from typing import Optional
from datetime import datetime


def is_before(date: str, cutoff: str) -> bool:
    d1 = datetime.strptime(date, "%Y-%m-%d")
    d2 = datetime.strptime(cutoff, "%Y-%m-%d")
    return (d1 - d2).total_seconds() < 0


def load_data(
    results_dir, benchmark, metric: str = "plausible", cutoff_date: Optional[str] = None
):
    with open(os.path.join(results_dir, "../models.json")) as f:
        model_list = json.load(f)["models"]
    rows = []

    llm_name, strategy, provider = model_list[-1]
    all_tests = set()
    for llm_name, strategy, provider in model_list:
        try:
            all_tests = all_tests.union(
                os.listdir(os.path.join(results_dir, llm_name, benchmark, "patches"))
            )
        except:
            pass
    print(f"benchmark {benchmark} contains {len(all_tests)} tests")

    if cutoff_date is not None:
        with open(os.path.join(results_dir, "release_dates.json")) as fd:
            model2date = json.load(fd)
            model_list = [
                el for el in model_list if is_before(model2date[el[0]], cutoff_date)
            ]

    for llm_name, strategy, provider in model_list:
        try:
            with open(
                os.path.join(
                    results_dir,
                    llm_name,
                    benchmark,
                    f"statistics_{benchmark}_instruct_{strategy}.json",
                )
            ) as fd:
                stats = json.load(fd)
                valid_tests = stats[f"bugs_with_{metric}_candidates"]
                for test in valid_tests:
                    rows.append((llm_name, test, 1))
                for test in set(all_tests).difference(valid_tests):
                    rows.append((llm_name, test, 0))
        except FileNotFoundError:
            print(f"{llm_name} has no data for {benchmark}")

    return rows


# Load the data
cutoff_date = None
if len(sys.argv) > 2:
    cutoff_date = sys.argv[2]
benchmarks = ["defects4j", "gitbugjava"]
for benchmark in benchmarks:
    results_dir = os.path.join(sys.argv[1], "results")
    data = load_data(results_dir, benchmark, cutoff_date=cutoff_date)
    filename = (
        f"./{benchmark}.csv"
        if cutoff_date is None
        else f"./{benchmark}_cutoff_{cutoff_date}.csv"
    )
    with open(filename, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["variant", "test", "success"])
        writer.writerows(data)

# print(data)
# # Create a DataFrame
# df = pd.DataFrame(data)
