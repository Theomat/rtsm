import argparse
from typing import List
import pandas as pd
from os import listdir
import os

import tqdm


###load one csv file specified by its path and filename
###the file is supposed to be a csv file that contains configurations of systems and performance measures over an execution
###return the number of line in the file and the data stored in the file
def load_csv(path, filename, with_name=True):
    filename = os.path.join(path, filename)
    data = pd.read_csv(filename, header=0)

    if with_name:
        data["inputname"] = os.path.splitext(os.path.basename(filename))[0]

    nb_data = data.shape[0]
    return data, nb_data


###find files in a specific directory with a specifc extension
###return a list of filename
def find_files(path, ext="csv"):
    ext = ext
    # list all the files in the folder
    filenames = listdir(path)
    # list files that have the specified extension
    filename_list = [filename for filename in filenames if filename.endswith(ext)]
    return filename_list


### load and return all the files with a specific extension that are in a specific directory
### return the number of line in the file and a dataframe containing all data stored in the files that have been found
def load_all_csv(path, ext="csv", with_names=True):
    files_to_load = find_files(path, ext)
    # load first data file alone
    all_data, nb_config = load_csv(path, files_to_load[0], with_name=with_names)
    # load the rest and append to the previous dataframe
    for f in tqdm.tqdm(files_to_load[1:], desc="loading data"):
        app_data, a = load_csv(path, f, with_name=with_names)
        all_data = pd.concat([all_data, app_data])
    # all_data = pd.concat([pd.read_csv(path+'/'+f) for f in files_to_load])

    return all_data, nb_config


def load_data(path: str):
    # store indexes of interest (retrieving performance values) to perform clustering and analyzes

    # load all data in a single dataframe
    ext = "csv"
    perf_matrix, nb_data = load_all_csv(path, ext)
    return perf_matrix


def save_for_metric(df: pd.DataFrame, metrics: List[str], cost: str, ignore: List[str]):
    allowed = df.columns.to_list()
    assert all(m in allowed for m in metrics)
    assert len(cost) == 0 or cost in allowed

    config_columns = [x for x in allowed if x not in metrics and x not in ignore]
    if len(cost) > 0:
        filename: str = f"./{name}_{'_'.join(metrics)}_cost_{cost}.csv"
    else:
        filename: str = f"./{name}_{'_'.join(metrics)}.csv"
    config_columns.remove("inputname")

    df["variant"] = df[config_columns].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )
    df = df.rename(
        columns={
            "inputname": "test",
        }
    )
    df = df[["variant", "test"] + metrics]
    df.to_csv(filename, index=False)
    if len(cost) > 0:
        with open(filename) as fd:
            lines = fd.readlines()
        costs = df.groupby("test")[cost].sum()
        costs = costs.mul(1 / len(metrics))
        costs.to_csv(filename, index=True)
        with open(filename, "a+") as fd:
            fd.write("=" * 80 + "\n")
            fd.writelines(lines)
    print("Saved to", filename)


### managing arguments
if __name__ == "__main__":
    # Define arguments for cli and run main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        help="The path to folder to find data to load",
        type=str,
    )
    parser.add_argument(
        "--cost", help="The cost metric to be used", type=str, default=""
    )
    parser.add_argument(
        "-i",
        "--ignore",
        nargs="*",
        help="The cost metric to be used",
        type=str,
    )
    parser.add_argument(
        "metrics",
        help="The performance metric to be used",
        nargs="+",
        type=str,
    )
    args = parser.parse_args()
    folder: str = args.folder
    cost: str = args.cost
    ignore = args.ignore or []

    name: str = os.path.basename(folder)
    metrics: List[str] = args.metrics

    for i in range(1, len(metrics) + 1):
        df = load_data(args.folder)
        save_for_metric(df, metrics[:i], cost, ignore + metrics[i:])
