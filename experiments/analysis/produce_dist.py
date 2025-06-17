# PROUT /// script
# dependencies = [
#   "pandas",
#   "seaborn",
#   "pltpublish",
#   "tqdm"
# ]
# ///
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pltpublish as pub
import seaborn as sns
from scipy.integrate import trapezoid as simpson
import numpy as np

ACCEPTED_SOLVERS = sorted(["bs", "MILP", "rs"])

WA, WB = 1, 1


def score(ratio: float, kendall: float) -> float:
    return ((1 - ratio) * WA + (kendall * 0.5 + 0.5) * WB) / (WA + WB)


if __name__ == "__main__":
    df = pd.read_csv("./global_benchmark.csv")
    df["score"] = ((1 - df["ratio"]) * WA + (df["kendall"] * 0.5 + 0.5) * WB) / (
        WA + WB
    )
    df["fraction"] = df["test"].str.split("_").str[-3]
    df = df[df["variant"].isin(ACCEPTED_SOLVERS)]
    df = df.sort_values(by=["variant"])

    stratified_data = {}

    pub.setup()
    colors = pub.get_color_cycle()
    c = colors[2]
    colors[2] = colors[0]
    colors[0] = c
    # test,variant,ratio,kendall
    for f in df["fraction"].unique():
        sub_df = df[df["fraction"] == f]

        ax = sns.displot(data=sub_df, x="score", hue="variant", kind="ecdf")
        ax = plt.gca()
        for line, c, solver in zip(ax.lines, colors, ACCEPTED_SOLVERS[::-1]):
            x = line.get_xydata()[:, 0]
            y = line.get_xydata()[:, 1]
            ax.fill_between(x, y, color=c, alpha=0.3)
            auc = simpson(y[1:], x[1:])
            print(f"{solver} AUC=", auc)
        # plt.vlines(0.5, 0, 0.5, colors=["red"], linestyles="dashed")
        # plt.hlines(0.5, 0, 0.5, colors=s["red"], linestyles="dashed")
        plt.xlabel("Score")
        # plt.xlim(left=0)
        plt.legend()
        plt.tight_layout()

        pub.save_fig(f"./cumulative_score_dist_{f}.png", scale=2)
        plt.show()
        plt.close()
