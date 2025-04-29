import sys
import os
import json

from aslib_scenario import ASlibScenario

path = sys.argv[1]
scenario = ASlibScenario()

scenario.read_scenario(path)
scenario.check_data()
name = os.path.basename(scenario.dir_)

with open(sys.argv[2]) as fd:
    data = json.load(fd)
solution_list = data["solutions"]
one_sol = solution_list[0]


print("Total instances:", len(scenario.instances))
print("Selected:", len(set(scenario.instances) & set(one_sol)), "/", len(one_sol))
feature_mask = scenario.feature_data.index.isin(one_sol)
all_features_df = scenario.feature_data
sol_features_df = scenario.feature_data[feature_mask]
print(all_features_df.describe())
print(sol_features_df.describe())
