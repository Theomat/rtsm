from rtsm.instance import Instance
import numpy as np
import time
from pulp import LpBinary, LpVariable, LpProblem, lpSum


def instance_to_milp(instance: Instance) -> str:
    start = time.perf_counter_ns()
    p = LpProblem()
    use_vars = [LpVariable(f"u{i}", cat=LpBinary) for i in range(len(instance.tests))]
    p.addVariables(use_vars)
    cst = lpSum(instance.costs[i] * use_vars[i] for i in range(len(instance.tests)))
    p.setObjective(cst)

    total = np.sum(instance.performance_matrix, axis=-1).reshape((-1,))
    for i in range(len(instance.variants)):
        for k in range(i + 1, len(instance.variants)):
            left_exp = lpSum(
                instance.performance_matrix[0, i, j] * use_vars[j]
                for j in range(len(instance.tests))
                if instance.performance_matrix[0, i, j] > 0
            )
            right_exp = lpSum(
                instance.performance_matrix[0, k, j] * use_vars[j]
                for j in range(len(instance.tests))
                if instance.performance_matrix[0, k, j] > 0
            )
            diff = left_exp - right_exp
            if total[i] <= total[k]:
                p.addConstraint(diff <= -1e-4, name=f"v_{i}_{k}")
            else:
                p.addConstraint(diff >= 1e-4, name=f"v_{i}_{k}")

    for var in use_vars:
        var.setInitialValue(1)
    p.solve()
    end = time.perf_counter_ns()

    out = {
        "solutions": [
            [t for i, t in enumerate(instance.tests) if use_vars[i].varValue > 0]
        ]
        if p.sol_status == 1
        else [instance.tests],
        "accuracy": 1.0,
        "predictor": "none",
        "base_solver": "MIP",
        "runtime": (end - start) / 1e9,
    }
    return out


if __name__ == "__main__":
    import argparse
    import sys
    import json
    from rtsm.utils.color_helper import get_color_helper

    F = get_color_helper()
    from rtsm.helper import (
        get_predictors,
        get_data_loaders,
        get_solvers,
        try_load_instance,
    )

    data_loaders, supported_extensions = get_data_loaders()
    predictors = get_predictors()
    solvers = get_solvers()

    parser = argparse.ArgumentParser(
        description="Ranked test suite minimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        help=f"file containing the data, supported extensions are: {', '.join(supported_extensions)}",
    )
    group = parser.add_argument_group("solving")
    group.add_argument("--swap", action="store_true", help="swap variants and tests")

    parser.add_argument("-q", "--quiet", action="store_true")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./rtsm_solutions.json",
        help="destination json file containing the solutions",
    )

    args = parser.parse_args()

    verbose: bool = not args.quiet
    swap: bool = args.swap
    # Check output file can be written to otherwise it is useless to compute but not being able to save
    try:
        with open(args.output, "w") as fd:
            pass
    except IOError as e:
        print(
            f"{F.RED}output file cannot be written to:{F.RESET}\n", e, file=sys.stderr
        )
        sys.exit(1)
    # Check then load data
    instance = try_load_instance(args.file, data_loaders, swap)
    # Build predictor

    sol = instance_to_milp(instance)
    with open(args.output, "w") as fd:
        json.dump(sol, fd)
