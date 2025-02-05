import json
from typing import Tuple
import tqdm
from pathlib import Path


def process_eval(data: dict, plus: bool = True) -> Tuple[bool, dict]:
    d = data["eval"]
    out = {}
    for problem, sol in d.items():
        out[problem] = []
        for i, row in enumerate(sol["base"]):
            if row[0] == "timed out" or row[0] == "thrown exception":
                relevant = [1 for _ in row[1]]
            else:
                mapper = (lambda x: x) if row[0] != "failed" else (lambda x: 1 - x)
                relevant = list(map(mapper, row[1]))
            if plus:
                if "plus" in sol and sol["plus"]:
                    new_row = sol["plus"][i]
                    if new_row[0] == "timed out" or new_row[0] == "thrown exception":
                        relevant += [1 for _ in new_row[1]]
                    else:
                        mapper = (
                            (lambda x: x)
                            if new_row[0] != "failed"
                            else (lambda x: 1 - x)
                        )
                        to_add = list(map(mapper, new_row[1]))
                        relevant += to_add
                else:
                    return False, None

            success = sum(relevant) == 0
            out[problem].append(success)
    return True, out


def load_data(path: str = "./results_plus", plus: bool = True) -> dict:
    directory = Path(path)

    # List files in the directory
    files = [file for file in directory.iterdir() if file.is_file()]
    data = {}
    for file in tqdm.tqdm(files, desc="loading data"):
        name = file.name
        index = name.index("_temp_")
        model_name = name[:index]
        end = len(name) - len(".json")
        if "_pass" in name:
            continue
        temp_str = name[index + len("_temp_") : end]
        if len(temp_str) == 0:
            continue
        temperature = float(temp_str)
        # print("loading:", model_name, "temperature:", temperature)
        with open(file) as fd:
            succ, d = process_eval(json.load(fd), plus)
            if succ:
                data[(model_name, temperature)] = d
    return data


def map_data_to_pass_k(data: dict, pass_k: int) -> dict:
    out = {}
    for name, val in data.items():
        out[name] = {}
        for key, sol in val.items():
            task_id = int(key[key.index("/") + 1 :])
            out[name][task_id] = any(sol[:pass_k])
    return out


if __name__ == "__main__":
    import argparse
    import sys
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=1, help="Pass K (default: 1)")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="load only base (not human eval plus) (default: False)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="./results",
        help="folder from which to load data (default: './results')",
    )

    params = parser.parse_args(sys.argv[1:])
    k = params.k
    base_only = params.base_only
    folder = params.folder

    data = load_data(folder, not base_only)
    data = map_data_to_pass_k(data, k)

    suffix = "" if base_only else "_plus"

    with open(f"./humaneval{suffix}_pass{k}.csv", "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["variant", "test", "performance"])
        for (model, temp), val in data.items():
            for test, score in val.items():
                writer.writerow([f"{model}-{temp}", test, float(score)])
    print("Saved to", f"./humaneval{suffix}_pass{k}.csv")
