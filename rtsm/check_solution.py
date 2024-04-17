

if __name__ == "__main__":
    from typing import Callable, Dict
    import argparse
    import sys
    import os
    import json
    import time

    import numpy as np
    from colorama import Fore as F

    from rtsm.instance import Instance
    from rtsm.data_loaders.csv_data_loader import CSVDataLoader
    from rtsm.predictors.predictor import Predictor, to_ranking, ranking_error
    from rtsm.predictors.logistic_boolean_predictor import LogisticBooleanPredictor
    from rtsm.predictors.logistic_rank_predictor import LogisticRankPredictor
    from rtsm.predictors.linear_predictor import LinearRegressionPredictor

    # Data Loaders
    data_loaders = [CSVDataLoader()]
    supported_extensions = set()
    for loader in data_loaders:
        supported_extensions.update(loader.get_extensions())

    # Predictors
    def adaptative_predictor(inst: Instance, **kwargs) -> Predictor:
        if np.unique(inst.performance_matrix).shape[0] == 2:
            return LogisticBooleanPredictor(inst, **kwargs)
        return LinearRegressionPredictor(instance, **kwargs)

    predictors: Dict[str, Callable[[Instance], Predictor]] = {
        "auto": adaptative_predictor,
        "logistic-bool": LogisticBooleanPredictor,
        "logistic-rank": LogisticRankPredictor,
        "linear": LinearRegressionPredictor,
    }

    # Data Loaders
    data_loaders = [CSVDataLoader()]
    supported_extensions = set()
    for loader in data_loaders:
        supported_extensions.update(loader.get_extensions())

    parser = argparse.ArgumentParser(
        description="Compare performances",
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
        "--predictor",
        choices=list(predictors.keys()),
        default=list(predictors.keys())[0],
        help=f"prediction model to use",
    )

    args = parser.parse_args()
    swap: bool = args.swap
    initial_solution: str = args.solution or ""

    # Check then load data
    loaders = [d for d in data_loaders if d.match(args.file)]
    if len(loaders) == 0:
        print(f"{F.RED}file format not supported:{F.RESET}", args.file, file=sys.stderr)
        sys.exit(1)
    instance = loaders.pop(0).load(args.file)
    # Swap instance
    if swap:
        instance = instance.swap()
    print(
        f"loaded {F.CYAN}{len(instance.variants)}{F.RESET} variants and {F.CYAN}{len(instance.tests)}{F.RESET} tests"
    )
    if not instance.check_filled():
        print(
            f"{F.YELLOW}warning:{F.RESET} the performance matrix is not completely filled!"
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
        solution_list = json.load(fd)
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
    predictor: Predictor = predictors[args.predictor](instance)
    print(f"prediction model: {F.CYAN}{predictor.get_name()}{F.RESET}")


    R = to_ranking(np.sum(instance.performance_matrix, axis=1))
    sR = to_ranking(np.sum(instance.subset(one_sol).performance_matrix, axis=1))
    print(
        f"ranking error without prediction: {F.GREEN}{ranking_error(R, sR):.2%}{F.RESET}"
    )
    start = time.perf_counter()
    error = predictor.ranking_error([x in one_sol for x in instance.tests])
    duration = time.perf_counter() - start
    print(
        f"ranking error with prediction ({F.LIGHTYELLOW_EX}{duration:.2f}{F.RESET}s): {F.GREEN}{error:.2%}{F.RESET}"
    )
