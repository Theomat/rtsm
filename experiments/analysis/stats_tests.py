# /// script
# dependencies = [
#   "pandas",
#   "scipy",
#   "numpy",
# ]
# ///
import glob
import sys
import numpy as np
import pandas as pd
import os
from scipy.stats import wilcoxon
from multiprocessing import Pool


folder = sys.argv[1]
dst = "./stats/"


SOLVERS = ["bs", "rs", "pca", "greedy", "MILP", "variance", "friedman"]
THRESHOLD = 0.05
FRACTIONS = (25, 50, 75, 100)
WA = 1
WB = 1


def stat_test(file: str):
    filename = os.path.basename(file)[: -len(".csv")]
    df = pd.read_csv(file)
    solvers = sorted(df["solver"].unique().tolist())
    count_matrix = np.zeros((len(SOLVERS), len(SOLVERS), len(FRACTIONS) + 1))
    df["ratio"] = df["cost"] / df["total_cost"]
    df["score"] = ((1 - df["ratio"]) * WA + (df["kendall"] * 0.5 + 0.5) * WB) / (
        WA + WB
    )
    scores = {s: {f: [] for f in FRACTIONS} for s in solvers}
    all_scores = {s: [] for s in solvers}
    data = []
    for group_name, group_df in df.groupby(["seed", "partition_seed", "fraction"]):
        fraction = group_name[-1]
        key = f"{filename}_{fraction}_{group_name[0]}_{group_name[1]}"
        if "MILP" not in group_df["solver"].unique():
            solver = "MILP"
            pot_df = df[
                (df["fraction"] == fraction)
                & (df["partition_seed"] == group_name[1])
                & (df["solver"] == solver)
            ][["score", "ratio", "kendall"]]
            if len(pot_df) == 1:
                score = pot_df.iloc[0]["score"]
                scores[solver][fraction].append(float(score))
                all_scores[solver].append(float(score))
                data.append(
                    (
                        key,
                        solver,
                        float(pot_df.iloc[0]["ratio"]),
                        float(pot_df.iloc[0]["kendall"]),
                    )
                )
            else:
                print(
                    "no info on MILP for:",
                    file,
                    "with fraction=",
                    fraction,
                    "partition_seed=",
                    group_name[1],
                )

        for row in group_df[["solver", "score", "ratio", "kendall"]].to_dict("split")[
            "data"
        ]:
            solver, score = row[0], row[1]
            scores[solver][fraction].append(float(score))
            all_scores[solver].append(float(score))
            data.append((key, solver, row[2], row[3]))

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
                if len(scores[s2][f]) <= 0:
                    print(f"[{file}] no data for", s1, "vs", s2, "fraction", f)
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
    return count_matrix, all_scores, data


if __name__ == "__main__":
    files = list(glob.glob(f"{folder}/*.csv"))
    with Pool() as p:
        result = p.map(stat_test, files)
    cmp_matrix = None
    all_scores = {s: [] for s in SOLVERS}
    data = []

    n = 0

    for add, dico, d in result:
        data += d
        for s in SOLVERS:
            all_scores[s] += dico[s]
        if cmp_matrix is None:
            cmp_matrix = add
        else:
            cmp_matrix += add
        n += 1

    with open("./global_benchmark.csv", "w") as fd:
        fd.write("test,variant,ratio,kendall\n")
        fd.write("\n".join(map(lambda x: ",".join(map(str, x)), data)))
    # print(cmp_matrix.tolist())
    print(n)
    for i, s1 in enumerate(SOLVERS):
        for j, s2 in enumerate(SOLVERS):
            if j <= i:
                continue
            alts = []
            positives = np.sum(np.asarray(all_scores[s1]) >= np.asarray(all_scores[s2]))
            negatives = np.sum(np.asarray(all_scores[s1]) < np.asarray(all_scores[s2]))
            effect_size = (positives - negatives) / len(all_scores[s1])
            for alternative in [
                "two-sided",
                "greater",
                "less",
            ]:
                stat = wilcoxon(all_scores[s1], all_scores[s2], alternative=alternative)
                alts.append(float(stat.pvalue))
            print(f"{s1} vs {s2} (effect={effect_size}) = {alts}")
