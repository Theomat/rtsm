# /// script
# dependencies = [
#   "pandas",
#   "scipy",
#   "numpy",
#   "tqdm"
# ]
# ///
import glob
import sys
import numpy as np
import pandas as pd
import os
import tqdm
from scipy.stats import wilcoxon


folder = sys.argv[1]
dst = "./stats/"


SOLVERS = ["bs", "rs", "pca", "greedy", "MILP"]
THRESHOLD = 0.05
FRACTIONS = (25, 50, 75)


def stat_test(file: str):
    filename = os.path.basename(file)[: -len(".csv")]
    df = pd.read_csv(file)
    solvers = sorted(df["solver"].unique().tolist())
    count_matrix = np.zeros((len(SOLVERS), len(SOLVERS), len(FRACTIONS) + 1))
    df["ratio"] = df["cost"] / df["total_cost"]
    df["score"] = (1 - df["ratio"]) + df["kendall"] * 0.5 + 0.5
    scores = {s: {f: [] for f in FRACTIONS} for s in solvers}
    all_scores = {s: [] for s in solvers}
    for group_name, group_df in df.groupby(["seed", "partition_seed", "fraction"]):
        fraction = group_name[-1]
        for row in group_df[["solver", "score"]].to_dict("split")["data"]:
            solver, score = row[0], row[1]
            scores[solver][fraction].append(float(score))
            all_scores[solver].append(float(score))

    matrix = [
        [
            "solver1",
            "solver2",
            "better",
            "pvalue:two-sided",
            "pvalue:greater",
            "pvalue:less",
            "fraction",
        ]
    ]
    for i, s1 in enumerate(solvers):
        for s2 in solvers[i + 1 :]:
            # SCORES PER FRACTION
            for i, f in enumerate(FRACTIONS):
                if len(scores[s1][f]) != len(scores[s2][f]):
                    print("failed", s1, "vs", s2, "fraction", f)
                    continue
                better = np.sum(scores[s1][f] >= scores[s2][f])
                alts = []
                for alternative in [
                    "two-sided",
                    "greater",
                    "less",
                ]:
                    stat = wilcoxon(
                        scores[s1][f], scores[s2][f], alternative=alternative
                    )
                    alts.append(stat.pvalue)
                if alts[0] <= THRESHOLD:
                    i1 = SOLVERS.index(s1)
                    i2 = SOLVERS.index(s2)
                    if alts[1] <= THRESHOLD:
                        count_matrix[i1, i2, i] += 1
                    elif alts[2] <= THRESHOLD:
                        count_matrix[i2, i1, i] += 1

                matrix.append([s1, s2, better] + alts + [f])
            # SCORES GLOBAL
            if len(all_scores[s1]) != len(all_scores[s2]):
                print("failed", s1, "vs", s2, "global")
                continue
            better = np.sum(all_scores[s1] >= all_scores[s2])
            alts = []
            for alternative in [
                "two-sided",
                "greater",
                "less",
            ]:
                stat = wilcoxon(all_scores[s1], all_scores[s2], alternative=alternative)
                alts.append(stat.pvalue)
            if alts[0] <= THRESHOLD:
                i1 = SOLVERS.index(s1)
                i2 = SOLVERS.index(s2)
                if alts[1] <= THRESHOLD:
                    count_matrix[i1, i2, len(FRACTIONS)] += 1
                elif alts[2] <= THRESHOLD:
                    count_matrix[i2, i1, len(FRACTIONS)] += 1

            matrix.append([s1, s2, better] + alts + [-1])

    with open(f"./{dst}/{filename}_stat.csv", "w") as fd:
        fd.writelines(map(lambda x: ",".join(map(str, x)) + "\n", matrix))
    return count_matrix


cmp_matrix = None
n = 0
for file in tqdm.tqdm(glob.glob(f"{folder}/*.csv")):
    add = stat_test(file)
    if cmp_matrix is None:
        cmp_matrix = add
    else:
        cmp_matrix += add
    n += 1
print(cmp_matrix.tolist())
print(n)
