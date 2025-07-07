from collections import defaultdict
import json
import sys
import glob

COLUMNS = "fraction,partition_seed,seed,solver,target_kendall,kendall,spearman,size,runtime,cost,total_cost,variants,tests".split(
    ","
)


def is_float(x) -> bool:
    try:
        float(x)
    except ValueError:
        return False
    return True


def get_name(idx) -> str:
    return COLUMNS[idx]


def fix_size_and_runtime(file: str) -> tuple[int, float]:
    with open(file.replace(".tmp", ".json")) as fd:
        data = json.load(fd)
        runtime = float(data["runtime"])
        size = len(list(data["solutions"]).pop())
    return size, runtime


def read_file(file: str) -> dict:
    parts = [x for x in file.split(".")[:-1] if len(x) > 0]
    filename = parts[0]
    filename = filename[filename.rfind("/") + 1 :]

    data = []
    fraction = parts[-4]
    data.append(fraction if fraction != "5" else "50")
    data.append(parts[-5])
    data.append(parts[-3])
    data.append(parts[-2])
    data.append(parts[-1])
    with open(file) as fd:
        lines = [x.strip("\n ") for x in fd.readlines() if len(x.strip("\n ")) > 0]
        if len(lines) <= 2:
            print("skipping:", file)
            return {}
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

        chars = "".join([chr(83), chr(0)])
        spearman = worst
        data.append(kendall)
        data.append(spearman)
        data.append(lines[-1].strip(chars))  # size
        data.append(lines[-2].strip(chars))  # runtime
        if "\t" in data[-1] or "\t" in data[-2]:
            # We are missing size or runtime so fix this
            size, runtime = fix_size_and_runtime(file)
            data[-2] = size
            data[-1] = runtime
        cost = [
            line[line.index(":") + 1 :].strip() for line in lines if "cost:" in line
        ][0]
        sol_cost = cost[: cost.index("/")].strip()
        total_cost = cost[cost.index("/") + 1 : cost.index("(")].strip()
        data.append(sol_cost)
        data.append(total_cost)
        parts = lines[0].split(" ")
        data.append(parts[1])
        data.append(parts[4])
        for i in range(-8, 0):
            assert is_float(data[i]), (
                f"failed parsing [{get_name(i)}]:{file} [{i}]= <<{data[i]}>> :{ord(data[i][0])}"
            )

    return {"file": filename, "data": data}


if __name__ == "__main__":
    import os
    import csv
    import tqdm

    folder = sys.argv[1]
    DST = sys.argv[2] if len(sys.argv) >= 3 else "."

    os.makedirs(DST, exist_ok=True)

    content = defaultdict(list)
    skipped = 0
    for file in tqdm.tqdm(glob.glob(f"{folder}/*.tmp")):
        dico = read_file(file)
        if len(dico) == 0:
            skipped += 1
            continue
        file = dico["file"] + ".csv"
        content[file].append(dico["data"])
    print(skipped, "files skipped because solution check failed!")
    for file, lines in content.items():
        with open(os.path.join(DST, file), "w") as fd:
            writer = csv.writer(fd)
            writer.writerow(COLUMNS)
            writer.writerows(lines)
