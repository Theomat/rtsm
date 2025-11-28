import glob
import os
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.patches as mpatches
import pltpublish as pub

pub.setup()

dst_folder = "plots"

KENDALL= 1
SOLVERS = ["bs", "rs", "pca", "greedy", "MILP"]
FRACTION = 100

def break_even(csv_path, target_kendall=KENDALL, fraction=FRACTION):
    df = pd.read_csv(csv_path)

    # Filter only rows matching target_kendall and fraction
    df = df[(df["target_kendall"] == target_kendall) & (df["fraction"] == fraction)]

    if df.empty:
        print("No rows found for this target_kendall and fraction.")
        return {s: [] for s in SOLVERS}

    df = df.copy()
    df = df[df["delta_cost"].astype(float) > 0]  # avoid division issues

    if df.empty:
        print(f"All rows have non-positive delta cost. Cannot compute break-even for {csv_path}.")
        return {s: [] for s in SOLVERS}

    df["break_even"] = df["runtime"] / df["delta_cost"]
    return  (
        df.groupby("solver")["break_even"]
        .apply(list)
        .to_dict()
    )



def plot_bep(bep):
    # Make sure each variant has a list (maybe empty) for consistent ordering
    data_per_variant = [bep.get(v, []) for v in SOLVERS]

    # Skip fractions where no data is available at all
    if all(len(d) == 0 for d in data_per_variant):
        return

    fig, ax = plt.subplots()
    print(data_per_variant)
    # Matplotlib’s violinplot positions are 1..N
    parts = ax.violinplot(
        data_per_variant,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    
    )

    colors = pub.get_color_cycle()
    violin_color = "#87CEFA"
    mean_color = colors[3]
    median_color = "purple"

    for pc in parts["bodies"]:
        vertices = pc.get_paths()[0].vertices
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        # Create new polygons
        verts_above = vertices

        # Remove the original body
        pc.set_alpha(0)

        # Plot above-half in blue
        poly_above = PolyCollection(
            [verts_above],
            facecolor=violin_color,
            edgecolor="black",
            alpha=0.75,
        )
        ax.add_collection(poly_above)

    # Mean line
    # if "cmeans" in parts:
    #     parts["cmeans"].set_color("purple")
    #     parts["cmeans"].set_linewidth(2)

    # Median line
    if "cmedians" in parts:
        parts["cmedians"].set_color(median_color)
        parts["cmedians"].set_linewidth(2)
        parts["cmedians"].set_linestyle("--")

        # ------------------------------------------------------------
    # ADD Q1 AND Q3 (25th & 75th percentiles)
    # ------------------------------------------------------------

    for i, scores in enumerate(data_per_variant):
        if len(scores) == 0:
            continue

        x = i + 1  # position of violin
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)

        # Q1 line
        ax.hlines(
            q1, x - 0.2, x + 0.2, colors=median_color, linewidth=1.5, linestyle=":"
        )

        # Q3 line
        ax.hlines(
            q3, x - 0.2, x + 0.2, colors=median_color, linewidth=1.5, linestyle=":"
        )
    # ------------------------------------------------------------
    # ADD LEGEND
    # ------------------------------------------------------------
    legend_handles = [
        # mpatches.Patch(color=mean_color, label="Mean"),
        mpatches.Patch(color=median_color, label="Median, Q1 and Q3"),
    ]
    ax.legend(handles=legend_handles)

    # ------------------------------------------------------------
    # AXES / TITLE
    # ------------------------------------------------------------
    ax.set_xticks(range(1, len(SOLVERS) + 1))
    ax.set_xticklabels([mapping.get(x, x) for x in SOLVERS])
    ax.set_xlabel("Method")
    ax.set_ylabel("Break Even Point")
    # ax.set_title(f"Violin plot – target={KENDALL}, fraction={fraction}")

    fig.tight_layout()
    pub.save_fig(
        os.path.join(dst_folder, f"bep_{KENDALL}_frac{FRACTION}.png"),
        scale=2,
    )
    plt.close(fig)


if __name__ == "__main__":
    bep = {solver: [] for solver in SOLVERS}

    plt.rcParams["text.usetex"] = True

    mapping = {
        "bs": "BISS",
        "MILP": "MILP",
        "rs": "RANDOM",
        "variance": ("GREEDY"),
        "pca": ("PCA"),
    }

    total = 0
    files = [file for file in glob.glob("csv/*.csv") if str(file).endswith("time.csv")]
    print("Kept", len(files), "benchmarks:", files)
    with Pool() as p:
        result = p.map(break_even, files)
    for dico in result:
        for s in SOLVERS:
            bep[s] += dico.get(s, [])
            # if timeouts > 0:
            #     files[s].append(file)
    plot_bep(bep)


