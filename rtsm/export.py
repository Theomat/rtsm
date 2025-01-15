if __name__ == "__main__":
    import argparse
    import sys
    import os
    import json

    from colorama import Fore as F

    from rtsm.predictors.predictor import Predictor
    from rtsm.helper import get_predictors, get_data_loaders, try_load_instance

    # Data Loaders
    data_loaders, supported_extensions = get_data_loaders()
    predictors = get_predictors()

    parser = argparse.ArgumentParser(
        description="Export prediction model for specific solution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        help=f"file containing the data, supported extensions are: {', '.join(supported_extensions)}",
    )
    parser.add_argument(
        "solution",
        type=str,
        help="the solution to compare with",
    )
    parser.add_argument("--swap", action="store_true", help="swap variants and tests")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./model.json",
        help="destination json file containing the prediction model",
    )
    args = parser.parse_args()
    swap: bool = args.swap
    initial_solution: str = args.solution or ""

    # Check then load data
    instance = try_load_instance(args.file, data_loaders, swap)
    print(
        f"loaded {F.CYAN}{len(instance.variants)}{F.RESET} variants and {F.CYAN}{len(instance.tests)}{F.RESET} tests"
    )
    # Load solution
    one_sol = []
    if not os.path.exists(initial_solution) or not os.path.isfile(initial_solution):
        print(
            f"{F.RED}solution file does not exist or is not a valid file:{F.RESET}",
            initial_solution,
            file=sys.stderr,
        )
        sys.exit(1)
    with open(initial_solution) as fd:
        data = json.load(fd)
        predictor_name = data["predictor"]
        solution_list = data["solutions"]
        if len(solution_list) == 0:
            print(
                f"{F.RED}solution file contains no solution!{F.RESET}",
                file=sys.stderr,
            )
            sys.exit(1)
        one_sol = solution_list[0]
        print(
            f"loaded solution of size {F.CYAN}{len(one_sol)}{F.RESET} ({F.CYAN}{len(one_sol)/len(instance.tests):.1%}{F.RESET})"
        )
    # Build predictor
    predictor: Predictor = predictors[predictor_name](instance)
    print(f"prediction model: {F.CYAN}{predictor.get_name()}{F.RESET}")

    predictor.export_prediction([x in one_sol for x in instance.tests]).export(
        args.output
    )
    print(f"saved to: {F.CYAN}{args.output}{F.RESET}")
