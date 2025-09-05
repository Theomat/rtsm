import pandas as pd
import matplotlib.pyplot as plt
import pltpublish as pub
import seaborn as sns

plt.rcParams["text.usetex"] = True


if __name__ == "__main__":
    with open("./improvements.txt") as fd:
        data = map(float, fd.readlines())

    df = pd.DataFrame({"Crunch Loops": [x for x in data if x > 0]})

    pub.setup()
    ax = sns.displot(data=df, kind="ecdf")
    # plt.vlines(x=1, ymin=0, ymax=1, label="No crunching", colors=["red"])
    y = 0.23
    plt.annotate(
        r"76\% benefit from iterative solving",  # LaTeX-formatted text
        xy=(.95, y),  # Point to annotate (on curve)
        xytext=(1 + 3, y - .2),  # Text location
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        # fontsize=12,
    )
    plt.xlabel("Iterations&")
    plt.ylabel("Cumulative Distribution")
    ax.legend.set_visible(False)
    # ax.legend.
    plt.tight_layout()
    plt.grid()
    pub.save_fig(f"./crunch_loops.png", scale=2)
    plt.show()
    # plt.close()

