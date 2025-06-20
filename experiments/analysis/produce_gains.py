from collections import defaultdict
import csv
import numpy as np

__HEADLINE__ = "Fraction & \\multicolumn{1}{l}{MILP} & \\multicolumn{1}{l}{\\prefixours} & \\multicolumn{1}{l}{\\prefix{PCA}} & \\multicolumn{1}{l}{\\prefix{random}} & \\multicolumn{1}{l}{\\prefix{greedy}}"

WA, WB = 1, 1
__PART1 = """{\\centering \n
    \\begin{longtable}{@{}r|llllll@{}}
        \\toprule \\\\ """
__PART2 = """\\\\\n\\midrule \\\\\n"""
__PART3 = """\\\\\n\\bottomrule
    \\caption{"""


TO_REMOVE = set(["friedman", "greedy"])


def make_template(headline, content, capt_name, label) -> str:
    return (
        __PART1
        + headline
        + __PART2
        + content
        + __PART3
        + capt_name
        + "}\\label{table:"
        + label
        + "}\n\\end{longtable}\n}"
    )


def score(ratio: float, kendall: float) -> float:
    return 1 - ratio


def rename_filename(filename: str) -> str:
    name = filename.replace("_", "-")
    for i in range(2, 5):
        name = name.replace(f"-q{i}", "")
    for i in range(6, 10):
        name = name.replace(f"-q{i}", "")
    for i in range(11, 15):
        name = name.replace(f"-q{i}", "")
    if "-cost" in name:
        name = name[: name.find("-cost")]

    return name


def to_table(
    dico: dict[str, dict[str, list[tuple[float, float]]]],
) -> str:
    capt_name = "Mean Cost reduction of different methods with all variants with 95\\% confidence interval in parenthesis if greater than 0. Statistically best performing methods are in \\textbf{bold}."
    content = ""
    fractions = []
    totals = defaultdict(list)
    for filename in sorted(dico.keys(), key=lambda x: x.lower()):
        name = rename_filename(filename)

        fraction_elems = []
        values = []
        solver2index = {}
        for solver, data in dico[filename].items():
            values.append((np.mean(data), 1.95 * np.std(data)))
            solver2index[solver] = len(values) - 1
            totals[solver] += data
        maxi = np.max([x[0] for x in values])
        best_index = [i for i in range(len(values)) if values[i][0] >= maxi].pop()
        for mean, std in values:
            is_bold = mean + std >= maxi - values[best_index][1]
            txt = f"{mean:.2f} ({std:.2f})".replace("(0.00)", "").strip()
            if is_bold:
                txt = "\\textbf{" + txt + "}"
            fraction_elems.append(txt)
        assert len(fraction_elems) == len(solver2index)
        assert len(fraction_elems) == 7
        print(sorted([s for s in sorted(solver2index.keys()) if s not in TO_REMOVE]))
        fractions.append(
            " & ".join(
                [name]
                + [
                    fraction_elems[solver2index[s]]
                    for s in sorted(solver2index.keys())
                    if s not in TO_REMOVE
                ]
            )
        )

    # Global
    fraction_elems = {}
    dico = {solver: (np.mean(x), 1.95 * np.std(x)) for solver, x in totals.items()}
    values = list(dico.values())
    maxi = np.max([x[0] for x in values])
    best_index = [i for i in range(len(values)) if values[i][0] >= maxi].pop()
    for key, (mean, std) in dico.items():
        is_bold = mean + std >= maxi - values[best_index][1]
        txt = f"{mean:.2f} ({std:.2f})".replace("(0.00)", "").strip()
        # if is_bold:
        # txt = "\\textbf{" + txt + "}"
        fraction_elems[key] = txt
    fractions.append("\\midrule")
    fractions.append(
        " & ".join(
            ["Average"]
            + [fraction_elems[s] for s in sorted(totals.keys()) if s not in TO_REMOVE]
        )
    )
    content = "\\\\\n".join(fractions)

    out = make_template(__HEADLINE__, content, capt_name, "gain")
    return out


if __name__ == "__main__":
    with open("./global_benchmark.csv") as fd:
        rows = [x for x in csv.reader(fd)]
        rows.pop(0)

    stratified_data = {}

    # test,variant,ratio,kendall
    for row in rows:
        test_parts = row[0].split("_")
        #  test = f"{filename}_{fraction}_{seed}_{partition_seed}"
        filename = "_".join(test_parts[:-3])
        fraction = float(test_parts[-3])
        if fraction < 100:
            continue
        seed = float(test_parts[-2])
        partition_seed = float(test_parts[-1])
        variant = row[1]
        ratio = float(row[2])
        kendall = float(row[3])

        if filename not in stratified_data:
            stratified_data[filename] = {}
        if variant not in stratified_data[filename]:
            stratified_data[filename][variant] = []

        stratified_data[filename][variant].append(score(ratio, kendall))

    tables = to_table(stratified_data)
    # sorted_tables = [x[1] for x in sorted(tables)]
    with open("./tables_gain.tex", "w") as fd:
        fd.write(tables)
