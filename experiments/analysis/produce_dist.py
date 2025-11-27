# /// script
# dependencies = [
#   "pandas",
#   "seaborn",
#   "pltpublish",
#   "scipy"
# ]
# ///
import pandas as pd
import matplotlib.pyplot as plt
import pltpublish as pub
import seaborn as sns
from scipy.integrate import trapezoid as simpson

ACCEPTED_SOLVERS = sorted(["bs", "MILP", "rs"])

plt.rcParams["text.usetex"] = True


def prefix(name: str) -> str:
    return r"\MakeUppercase{" + name + r"}"


mapping = {
    "bs": "BISS",
    "MILP": "MILP",
    "rs": "RANDOM",
}
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
    # KENDALL = 99 UNCOMMENT
    # df = df[df["test"].str.contains("_99_")]

    df["method"] = df["variant"].replace(mapping)

    df = df.sort_values(by=["method"])

    stratified_data = {}

    pub.setup()
    colors = pub.get_color_cycle()
    c = colors[2]
    colors[2] = colors[0]
    colors[0] = c
    # test,variant,ratio,kendall
    for f in df["fraction"].unique():
        sub_df = df[df["fraction"] == f]
        ax = sns.displot(data=sub_df, x="score", hue="method", kind="ecdf")
        ax.legend.set_frame_on(True)
        ax = plt.gca()
        for line, c, solver in zip(ax.lines, colors, ACCEPTED_SOLVERS[::-1]):
            x = line.get_xydata()[:, 0]
            y = line.get_xydata()[:, 1]
            ax.fill_between(x, y, color=c, alpha=0.3)
            auc = simpson(y[1:], x[1:])
        plt.xlabel("Score")
        plt.tight_layout()

        pub.save_fig(f"./cumulative_score_dist_{f}.png", scale=2)
        plt.show()
        plt.close()
