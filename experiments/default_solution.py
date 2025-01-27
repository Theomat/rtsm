if __name__ == "__main__":
    import argparse
    import sys
    import os
    import json
    import time

    from rtsm.utils.color_helper import get_color_helper

    F = get_color_helper()
    from rtsm.instance import Instance
    from rtsm.solution import Solution

    from rtsm.predictors.predictor import Predictor
    from rtsm.helper import (
        get_predictors,
        get_data_loaders,
        get_solvers,
        try_load_instance,
    )

    from rtsm.utils.argparse_helper import bounded_float

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
    group.add_argument(
        "--predictor",
        choices=list(predictors.keys()),
        default=list(predictors.keys())[0],
        help="prediction model to use",
    )
    group.add_argument(
        "--solver",
        choices=list(solvers.keys()),
        default=list(solvers.keys())[0],
        help="solver to use",
    )
    group.add_argument("--swap", action="store_true", help="swap variants and tests")

    group = parser.add_argument_group("sampling based solvers (rs, bisect, ga)")
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed used for probabilistic solvers",
    )

    group = parser.add_argument_group("approximate (linear, logistic-rank)")
    group.add_argument(
        "--kendall",
        type=bounded_float(0, 1),
        default=1.0,
        help="Kendall coefficient of the ranking needed",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./rtsm_solutions.json",
        help="destination json file containing the solutions",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        help="time to register",
    )

    args = parser.parse_args()

    swap: bool = args.swap
    runtime: float = args.timeout
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
    accuracy: float = args.kendall
    # Get solver
    solver = solvers[args.solver]

    start = time.perf_counter_ns()

    def save(sols):
        with open(args.output, "w") as fd:
            json.dump(
                {
                    "solutions": Solution.to_json(sols),
                    "runtime": runtime,
                    "base_solver": args.solver,
                    "predictor": args.predictor,
                    "accuracy": accuracy,
                },
                fd,
            )

    # Save solution
    save([Solution(instance, instance.tests)])
