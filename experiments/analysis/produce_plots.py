import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pltpublish as pub
import seaborn as sns
import os
import tqdm

from paretoset import paretoset

folder = sys.argv[1]
dst = sys.argv[2]
pub.setup()


def auto_save_fig(file: str):
    filename = os.path.basename(file)[: -len(".csv")]
    df = pd.read_csv(file)
    df["ratio"] = df["cost"] / df["total_cost"]
    df = df.sort_values(by=["solver"])
    # fraction,partition_seed,seed,solver,target_kendall,kendall,spearman,size,runtime,cost,total_cost
    for f in df["fraction"].unique():
        sub_df = df[df["fraction"] == f]
        mask = paretoset(sub_df[["ratio", "kendall"]].to_numpy(), sense=["min", "max"])
        pts = sub_df[["ratio", "kendall"]].to_numpy()[mask]
        indices = np.argsort(pts[:, 1])
        sns.jointplot(
            df[df["fraction"] == f], x="ratio", y="kendall", hue="solver", alpha=0.7
        )
        plt.plot(
            pts[indices, 0],
            pts[indices, 1],
            label="Pareto Front",
            c="r",
            linestyle="dotted",
        )
        plt.xlim(0, 1)
        plt.ylim(top=1)
        plt.xlabel("Cost Ratio")
        plt.ylabel("Kendall")
        plt.legend()
        plt.tight_layout()

        pub.save_fig(f"./{dst}/{filename}_{f}.png", scale=2)
        # plt.show()
        plt.close()


for file in tqdm.tqdm(glob.glob(f"{folder}/*.csv")):
    auto_save_fig(file)