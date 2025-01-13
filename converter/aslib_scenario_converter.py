import sys, csv, os
from typing import Dict

from aslib_scenario import ASlibScenario

path = sys.argv[1]
scenario = ASlibScenario()

scenario.read_scenario(path)
scenario.check_data()


def from_scenario(scenario: ASlibScenario) -> Dict[str, Dict[str, float]]:
    """
    Extract the results dictionnary from an ASLibScenario.

    Parameters:
    -----------
    - scenario (ASlibScenario) - the scenario to extract the results from.

    Return:
    -----------
    A dictionnary containing as keys the name of the instances.
    The value for each instance is a dictionnary with keys the algorithm's name and value the time it took.
    """
    results: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(scenario.index_to_instance):
        results[name] = {}
        for algo in scenario.performance_data.columns:
            results[name][algo] = scenario.performance_data[algo].iloc[i]
    return results


name = os.path.basename(scenario.dir_)
data = from_scenario(scenario)
variants = scenario.performance_data.columns
for variant in variants:
    with open(f"./{name}_without_{variant}.csv", "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["variant", "test", "performance"])
        for test, val in data.items():
            for v, score in val.items():
                if v == variant:
                    continue
                writer.writerow([v, test, float(score)])
