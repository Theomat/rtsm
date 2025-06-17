import os
import csv
from multiprocessing import Pool
import numpy as np

# __HEADLINE__ = "Fraction & \\multicolumn{1}{l}{\\prefixours} & \\multicolumn{1}{l}{\\prefix{random}} & \\multicolumn{1}{l}{\\prefix{PCA}} & \\multicolumn{1}{l}{\\prefix{greedy}} & \\multicolumn{1}{l}{MILP}"

WA, WB = 1, 1
__PART1 = """\\begin{table}[htb]\n
    \\centering
    \\begin{tabular}{@{}r|lllll@{}}
        \\toprule \\\\ """
__PART2 = """\\\\\n\\midrule \\\\\n"""
__PART3 = """\\\\\n\\bottomrule
    \\end{tabular}
    \\caption{"""


FIGURES = True
ACCEPTED_SOLVERS = sorted(["bs", "MILP", "rs"])

__HEADLINE__ = "Fraction & " + " & \\multicolumn{1}{l}".join(
    map(lambda x: "{" + x + "}", ACCEPTED_SOLVERS)
)


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
        + "}\n\\end{table}"
    )


def score(ratio: float, kendall: float) -> float:
    return ((1 - ratio) * WA + (kendall * 0.5 + 0.5) * WB) / (WA + WB)


def to_table(
    arg: tuple[str, dict[float, dict[str, list[tuple[float, float]]]]],
) -> tuple[str, str]:
    filename, dico = arg
    label = filename.lower()
    capt_name = f"Scores of different methods for {filename.replace('_', '-')}"
    content = ""
    fractions = []
    for fraction in sorted(dico.keys(), reverse=True):
        fraction_elems = []
        values = []
        solver2index = {}
        for solver, data in dico[fraction].items():
            values.append((np.mean(data), 1.95 * np.std(data)))
            solver2index[solver] = len(values) - 1
        maxi = np.max([x[0] for x in values])
        best_index = [i for i in range(len(values)) if values[i][0] >= maxi].pop()
        for mean, std in values:
            is_bold = mean + std >= maxi - values[best_index][1]
            txt = f"{mean:.2f} ({std:.2f})"
            if is_bold:
                txt = "\\textbf{" + txt + "}"
            fraction_elems.append(txt)
        assert len(fraction_elems) == len(ACCEPTED_SOLVERS)
        assert len(fraction_elems) == len(solver2index)
        fractions.append(
            " & ".join(
                [str(int(fraction))]
                + [fraction_elems[solver2index[s]] for s in ACCEPTED_SOLVERS]
            )
        )
    # content = "\\\\\n \\midrule \\\\ \n".join(fractions)
    content = "\\\\\n".join(fractions)

    out = make_template(__HEADLINE__, content, capt_name, label)
    if FIGURES:
        out += add_figures(filename)
    return filename, out


def add_figures(
    filename: str,
) -> str:
    out = "\n\\begin{figure}[hb]\n"
    added = 0
    for fraction in [25, 50, 75, 100]:
        file = filename + f"_{fraction}.png"
        subfigure = """\\begin{subfigure}[b]{0.5\\linewidth}
    \\centering
    \\includegraphics[width=\\linewidth]{./plots/"""
        subfigure += file
        subfigure += (
            """} 
    \\caption{"""
            + f"{fraction}\\% variants"
            + """}
  \\end{subfigure}%\n"""
        )
        if os.path.exists(os.path.join("./plots", file)):
            out += subfigure
            added += 1
            if added == 2:
                out += "\\\\"
    out += "\\caption{" + filename.replace("_", "-") + "}\n"
    out += "\\end{figure}\n\n"

    return out


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


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

    tasks = [(filename, stratified_data[filename]) for filename in stratified_data]
    with Pool() as p:
        tables = p.map(to_table, tasks)
    sorted_tables = [x[1] for x in sorted(tables)]
    i = 0
    for t in batch(sorted_tables, 20):
        with open(f"./tables{i}.tex", "w") as fd:
            fd.write("\n".join(t))
        i += 1
