import sys
import os
import json
import numpy as np

from aslib_scenario import ASlibScenario

path = sys.argv[1]
scenario = ASlibScenario()

scenario.read_scenario(path)
scenario.check_data()
name = os.path.basename(scenario.dir_)


order = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

all_features_df = scenario.feature_data
all_agg = all_features_df.describe()
all_agg_safe = all_agg.replace(0.0, 1e-10)
# print("Total instances:", len(scenario.instances))

filename = sys.argv[2]
first_part = filename[: -len("1.bs.1.json")]

print(all_agg.shape)


all = []
for seed in range(1, 11):
    with open(first_part + str(seed) + ".bs.1.json") as fd:
        data = json.load(fd)
    solution_list = data["solutions"]
    one_sol = solution_list[0]

    # print("\tSelected:", len(set(scenario.instances) & set(one_sol)), "/", len(scenario.instances))
    feature_mask = scenario.feature_data.index.isin(one_sol)
    sol_features_df = scenario.feature_data[feature_mask]

    rchange = (all_agg - sol_features_df.describe()).abs() / all_agg_safe
    # rchange.fillna(0, inplace=True)
    agg = rchange.agg(["median"], axis=1)
    out = [0 for x in order]
    for index, row in agg.iterrows():
        i = order.index(index)
        val = row["median"]
        out[i] = val
    all.append(out)

values = []
for meta in [min, np.median, np.mean, max]:
    local = []
    for i in range(len(order)):
        data = [x[i] for x in all]
        val = meta(data)

        local.append(f"{val:.2%}".replace("%", "\\%"))
    values.append(local)
        # mini = min(x[i] for x in all)
        # maxi = max(x[i] for x in all)

        # for x in all:
        #     v = x[i]
        #     x[i] = f"{v:.2%}".replace("%", "\\%")
        #     if v <= mini:
        #         x[i] = "\\underline{" + x[i] + "}"

        #     if v >= maxi:
        #         x[i] = "\\textbf{" + x[i] + "}"

for out in values:
    print(" & ".join(out) + "\\\\")
