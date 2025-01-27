from collections import defaultdict
import sys
import glob

COLUMNS = "fraction,partition_seed,seed,solver,target_kendall,kendall,spearman,size,runtime,cost,total_cost".split(
    ","
)


def read_file(file: str) -> dict:
    parts = file.split(".")[1:-1]
    filename = parts[0]
    filename = filename[filename.index("/") + 1 :]
    out = {}
    data = []
    data.append(parts[-4])
    data.append(parts[-5])
    data.append(parts[-3])
    data.append(parts[-2])
    data.append(parts[-1])
    with open(file) as fd:
        lines = fd.readlines()
        out["runtime"] = lines[-2]
        out["size"] = lines[-1]
        kendalls = [
            line[line.index(":") + 1 :].strip() for line in lines if "Kendall" in line
        ]
        kendall = kendalls[-1]
        spearmans = [
            line[line.index(":") + 1 :].strip() for line in lines if "Spearman" in line
        ][-1]
        worst = 1
        for part in spearmans.split("p-value:"):
            if ":" in part:
                val = float(part[part.index(":") + 1 :])
                worst = min(val, worst)

        spearman = worst
        data.append(kendall)
        data.append(spearman)
        data.append(lines[-1][:-1])
        data.append(lines[-2][:-1])
        cost = [
            line[line.index(":") + 1 :].strip() for line in lines if "cost:" in line
        ][0]
        sol_cost = cost[: cost.index("/")].strip()
        total_cost = cost[cost.index("/") + 1 : cost.index("(")].strip()
        data.append(sol_cost)
        data.append(total_cost)
    return {"file": file, "data": data}


if __name__ == "__main__":
    import os
    import csv
    import tqdm

    folder = sys.argv[1]
    DST = sys.argv[2] if len(sys.argv) >= 3 else "."

    os.makedirs(DST, exist_ok=True)

    content = defaultdict(list)

    for file in tqdm.tqdm(glob.glob(f"{folder}/*.tmp")):
        dico = read_file(file)
        file = dico["file"] + ".csv"
        content[file].append(dico["data"])

    for file, lines in content.items():
        with open(file, "w") as fd:
            writer = csv.writer(fd)
            writer.writerow(COLUMNS)
            writer.writerows(lines)
