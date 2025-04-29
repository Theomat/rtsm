import glob
import sys
import numpy as np
import pandas as pd
import os
import tqdm
from scipy.stats import wilcoxon


folder = sys.argv[1]
dst = "./stats/"


def stat_test(file: str):
    filename = os.path.basename(file)[: -len(".csv")]
    df = pd.read_csv(file)
    solvers = df["solver"].unique().tolist()
    df["ratio"] = df["cost"] / df["total_cost"]
    df["score"] = (1 - df["ratio"]) + df["kendall"] * 0.5 + 0.5
    scores = {s: [] for s in solvers}
    for group_name, group_df in df.groupby(["seed", "partition_seed", "fraction"]):
        for row in group_df[["solver", "score"]].to_dict("split")["data"]:
            solver, score = row[0], row[1]
            scores[solver].append(float(score))
    matrix = [
        [
            "solver1",
            "solver2",
            "better",
            "pvalue:two-sided",
            "pvalue:greater",
            "pvalue:less",
        ]
    ]
    for i, s1 in enumerate(solvers):
        for s2 in solvers[i + 1 :]:
            if len(scores[s1]) != len(scores[s2]):
                continue
            better = np.sum(scores[s1] >= scores[s2])
            alts = []
            for alternative in [
                "two-sided",
                "greater",
                "less",
            ]:
                stat = wilcoxon(scores[s1], scores[s2], alternative=alternative)
                alts.append(stat.pvalue)
            matrix.append([s1, s2, better] + alts)
    with open(f"./{dst}/{filename}_stat.csv", "w") as fd:
        fd.writelines(map(lambda x: ",".join(map(str, x)) + "\n", matrix))


for file in tqdm.tqdm(glob.glob(f"{folder}/*.csv")):
    stat_test(file)
