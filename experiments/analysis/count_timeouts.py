from collections import defaultdict
import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool


folder = sys.argv[1]
dst = "./stats/"


SOLVERS = ["bs", "rs", "pca", "variance"]  # , "MILP"]
THRESHOLD = 0.05
FRACTIONS = (25, 50, 75, 100)
TIMEOUT = 50 * 60


def count_timeouts(file: str):
    df = pd.read_csv(file)
    solvers = sorted(df["solver"].unique().tolist())
    df["timeout"] = df["runtime"].astype(float) >= TIMEOUT
    all_scores = {s: [] for s in solvers}
    n = 0
    data = {s: [] for s in solvers}
    for group_name, group_df in df.groupby(["seed", "partition_seed", "fraction"]):
        fraction = group_name[-1]
        if "MILP" not in group_df["solver"].unique():
            solver = "MILP"
            pot_df = df[
                (df["fraction"] == fraction)
                & (df["partition_seed"] == group_name[1])
                & (df["solver"] == solver)
            ][["timeout", "tests", "variants"]]
            if len(pot_df) == 1:
                score = pot_df.iloc[0]["timeout"]
                tests = pot_df.iloc[0]["tests"]
                variants = pot_df.iloc[0]["variants"]
                all_scores[solver].append(bool(score))
                data[solver].append((int(tests), int(variants), bool(score)))
            else:
                print(
                    "no info on MILP for:",
                    file,
                    "with fraction=",
                    fraction,
                    "partition_seed=",
                    group_name[1],
                )

        for row in group_df[["solver", "timeout", "tests", "variants"]].to_dict(
            "split"
        )["data"]:
            solver, score, tests, variants = row[0], row[1], row[2], row[3]
            all_scores[solver].append(bool(score))
            data[solver].append((int(tests), int(variants), bool(score)))
            n += 1

    return all_scores, n, data


if __name__ == "__main__":
    all_scores = {s: 0 for s in SOLVERS}
    all_rel = {s: [] for s in SOLVERS}
    files = {s: [] for s in SOLVERS}

    plt.rcParams["text.usetex"] = True

    def prefix(name: str) -> str:
        return r"DCC\textsubscript{\scriptsize\textsf{\MakeUppercase{" + name + r"}}}"

    mapping = {
        "bs": prefix("MBENCH"),
        "MILP": "MILP",
        "rs": prefix("random"),
        "variance": prefix("greedy"),
        "pca": prefix("pca"),
    }

    total = 0
    files = list(glob.glob(f"{folder}/*.csv"))
    with Pool() as p:
        result = p.map(count_timeouts, files)
    for dico, n, d in result:
        total += n
        for s in SOLVERS:
            timeouts = sum(dico[s])
            all_rel[s] += d[s]
            all_scores[s] += timeouts
            # if timeouts > 0:
            #     files[s].append(file)

    for solver, timeouts in all_scores.items():
        print(f"{solver} timeouts {timeouts}/{total} ({timeouts / total:.2%})")
        # print(f"\t{files[solver]}")

    for score_fn, name in [
        (lambda x: x[0], "tests"),
        (lambda x: x[1], "variants"),
        (lambda x: x[0] * x[0], "product"),
    ]:
        for solver, data in all_rel.items():
            per_value = defaultdict(int)
            total = defaultdict(int)
            for el in data:
                val = score_fn(el)
                per_value[val] += el[-1]
                total[val] += 1
            X = sorted(per_value.keys())
            Y = [per_value[x] / total[x] for x in X]
            plt.plot(X, Y, label=mapping[solver])
        # plt.title(name)
        plt.xlabel(name)
        plt.ylabel(r"\% Timeouts")
        plt.grid()
        plt.legend()
        plt.show()
