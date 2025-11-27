import os
import csv
from matplotlib.collections import PolyCollection
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import pltpublish as pub

pub.setup()

WA, WB = 1, 1
KENDALL = 1
ACCEPTED_SOLVERS = sorted(["bs", "MILP", "rs"])
# ACCEPTED_SOLVERS = sorted(["bs", "MILP", "rs", "pca", "greedy"])
dst_folder = "plots"
COST_REDUCTION = False

if COST_REDUCTION:
    def score(ratio: float, kendall: float) -> float:
        return 1 - ratio
else:
    def score(ratio: float, kendall: float) -> float:
        return ((1 - ratio) * WA + (kendall * 0.5 + 0.5) * WB) / (WA + WB)


mapping = {
    "bs": "BISS",
    "MILP": "MILP",
    "rs": "RANDOM",
    "pca": "PCA",
    "greedy": "GREEDY"
}


def violin_plot(aggregated, suffix=""):
        os.makedirs(dst_folder, exist_ok=True)

        fractions = sorted(aggregated.keys())
        variants = ACCEPTED_SOLVERS  # order on x-axis

        for fraction in fractions:
            # Make sure each variant has a list (maybe empty) for consistent ordering
            data_per_variant = [aggregated[fraction].get(v, []) for v in variants]

            # Skip fractions where no data is available at all
            if all(len(d) == 0 for d in data_per_variant):
                continue

            fig, ax = plt.subplots()
            # Matplotlib’s violinplot positions are 1..N
            parts = ax.violinplot(
                data_per_variant,
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )

            for pc in parts["bodies"]:
                vertices = pc.get_paths()[0].vertices
                xs = vertices[:, 0]
                ys = vertices[:, 1]

                # Split vertices into two masks
                threshold = -1 if COST_REDUCTION else .5
                above = ys >= threshold
                below = ys < threshold
                above = ys >= -9999
                below = ys < -9999

                # Create new polygons
                verts_above = vertices[above]
                verts_below = vertices[below]

                # Remove the original body
                pc.set_alpha(0)

                # Plot above-half in blue
                if len(verts_above) > 0:
                    poly_above = PolyCollection(
                        [verts_above],
                        facecolor="#87CEFA",
                        edgecolor="black",
                        alpha=0.75
                    )
                    ax.add_collection(poly_above)

                # Plot below-half in red
                if len(verts_below) > 0:
                    poly_below = PolyCollection(
                        [verts_below],
                        facecolor="red",
                        edgecolor="black",
                        alpha=0.75
                    )
                    ax.add_collection(poly_below)

            # Mean line
            if "cmeans" in parts:
                parts["cmeans"].set_color("purple")
                parts["cmeans"].set_linewidth(2)

            # Median line
            if "cmedians" in parts:
                parts["cmedians"].set_color("yellow")
                parts["cmedians"].set_linewidth(2)
                parts["cmedians"].set_linestyle("--")


                # ------------------------------------------------------------
            # ADD Q1 AND Q3 (25th & 75th percentiles)
            # ------------------------------------------------------------
            Q1_color = "orange"
            Q3_color = "orange"

            for i, scores in enumerate(data_per_variant):
                if len(scores) == 0:
                    continue

                x = i + 1  # position of violin
                q1 = np.percentile(scores, 25)
                q3 = np.percentile(scores, 75)

                # Q1 line
                ax.hlines(
                    q1, x - 0.2, x + 0.2, colors=Q1_color, linewidth=1.5, linestyle="--"
                )

                # Q3 line
                ax.hlines(
                    q3,
                    x - 0.2,
                    x + 0.2,
                    colors=Q3_color,
                    linewidth=1.5,
                    linestyle="--")
            # ------------------------------------------------------------
            # ADD LEGEND
            # ------------------------------------------------------------
            legend_handles = [
                mpatches.Patch(color="yellow", label="Median"),
                mpatches.Patch(color="purple", label="Mean"),
                mpatches.Patch(color=Q1_color, label="Q1 and Q3"),
            ]
            ax.legend(handles=legend_handles)

            # ------------------------------------------------------------
            # AXES / TITLE
            # ------------------------------------------------------------
            if not COST_REDUCTION:
                ax.axhline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.8)

            ax.set_xticks(range(1, len(variants) + 1))
            ax.set_xticklabels([mapping.get(x, x) for x in variants])
            ax.set_xlabel("Method")
            ax.set_ylabel("Cost Reduction" if COST_REDUCTION else "Score")
            # ax.set_title(f"Violin plot – target={KENDALL}, fraction={fraction}")

            fig.tight_layout()
            pub.save_fig(os.path.join(dst_folder, f"violin_target{KENDALL}_frac{fraction}{suffix}.png"), scale=2)
            plt.close(fig)


def aggregate_data(stratified_data):
    aggregated = {}
    for filename, frac_dict in stratified_data.items():
        for fraction, var_dict in frac_dict.items():
            if fraction not in aggregated:
                aggregated[fraction] = {}
            for variant, scores in var_dict.items():
                if variant not in aggregated[fraction]:
                    aggregated[fraction][variant] = []
                aggregated[fraction][variant].extend(scores)
    return aggregated

if __name__ == "__main__":
    with open("./global_benchmark.csv") as fd:
        rows = [x for x in csv.reader(fd)]
        rows.pop(0)

    stratified_data = {}

    # test,variant,ratio,kendall
    for row in rows:
        test_parts = row[0].split("_")
        #  test = f"{filename}_{target}_{fraction}_{seed}_{partition_seed}"
        filename = "_".join(test_parts[:-4])
        target = float(test_parts[-4])
        if target != KENDALL:
            continue
        fraction = float(test_parts[-3])
        seed = float(test_parts[-2])
        partition_seed = float(test_parts[-1])
        variant = row[1]
        if variant not in ACCEPTED_SOLVERS:
            continue
        ratio = float(row[2])
        kendall = float(row[3])

        if filename not in stratified_data:
            stratified_data[filename] = {}
        if fraction not in stratified_data[filename]:
            stratified_data[filename][fraction] = {}
        if variant not in stratified_data[filename][fraction]:
            stratified_data[filename][fraction][variant] = []

        stratified_data[filename][fraction][variant].append(score(ratio, kendall))

    # 1) Aggregate across all filenames (and therefore across all seeds / partition_seeds)
    #    so we get: aggregated[fraction][variant] = list of scores


    violin_plot(aggregate_data(stratified_data))
    if COST_REDUCTION:
        filtered_stratified = {}

        for filename, frac_dict in stratified_data.items():
            # Get ALL scores for this filename
            all_scores = []
            for fraction, var_dict in frac_dict.items():
                for variant, scores in var_dict.items():
                    all_scores.extend(scores)

            # Keep only filenames with strictly positive max score
            if len(all_scores) > 0 and max(all_scores) > 0:
                filtered_stratified[filename] = frac_dict
        print(len(filtered_stratified))
        violin_plot(aggregate_data(filtered_stratified), suffix="_improvement")


