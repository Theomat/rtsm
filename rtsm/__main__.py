if __name__ == "__main__":
    import argparse
    import sys
    import json
    from typing import List

    from colorama import Fore as F

    from rtsm.solution import Solution
    from rtsm.data_loaders.csv_data_loader import CSVDataLoader

    from rtsm.predictors.logistic_boolean_predictor import LogisticBooleanPredictor

    from rtsm.solvers.solver import Solver
    from rtsm.solvers.bisect_solver import BisectSolver
    from rtsm.solvers.rs_solver import RandomSamplingSolver

    # Data Loaders
    data_loaders = [CSVDataLoader()]
    supported_extensions = set()
    for loader in data_loaders:
        supported_extensions.update(loader.get_extensions())

    # Predictors
    predictors = {"logistic-bool": LogisticBooleanPredictor}

    # Solvers
    __solver_list__: List[Solver] = [BisectSolver(), RandomSamplingSolver()]
    solvers = {solver.get_name(): solver for solver in __solver_list__}

    parser = argparse.ArgumentParser(description="Ranked test suite minimisation")
    parser.add_argument(
        "file",
        help=f"file containing the data, supported extensions are: {', '.join(supported_extensions)}",
    )
    group = parser.add_argument_group("solving")
    group.add_argument(
        "--predictor",
        choices=list(predictors.keys()),
        default=list(predictors.keys())[0],
        help=f"prediction model to use, default: {list(predictors.keys())[0]}",
    )
    group.add_argument(
        "--solver",
        choices=list(solvers.keys()),
        default=list(solvers.keys())[0],
        help=f"solver to use, default: {list(solvers.keys())[0]}",
    )

    group = parser.add_argument_group("sampling algorithms (rs, bisect)")
    group.add_argument(
        "--samples", type=int, default=10000, help="number of samples, default: 10000"
    )

    parser.add_argument("-q", "--quiet", action="store_true")

    parser.add_argument(
        "-p",
        "--procs",
        type=int,
        default=1,
        help="number of processors to use, default: 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./rtsm_solutions.json",
        help="destination json file containing the solutions, default: ./rtsm_solutions.json",
    )

    args = parser.parse_args(sys.argv[1:])

    verbose: bool = not args.quiet
    procs: int = args.procs

    samples: int = args.samples

    # Check then load data
    loaders = [d for d in data_loaders if d.match(args.file)]
    if len(loaders) == 0:
        print(f"{F.RED}file format not supported:{F.RESET}", args.file, file=sys.stderr)
        sys.exit(1)
    instance = loaders.pop(0).load(args.file)
    if verbose:
        print(
            f"loaded {F.CYAN}{len(instance.variants)}{F.RESET} variants and {F.CYAN}{len(instance.tests)}{F.RESET} tests"
        )
        if not instance.check_filled():
            print(
                f"{F.YELLOW}warning:{F.RESET} the performance matrix is not completely filled!"
            )

    # Build predictor
    predictor = predictors[args.predictor](instance)
    if verbose:
        print(f"prediction model: {F.CYAN}{predictor.get_name()}{F.RESET}")

    # Get solver
    solver = solvers[args.solver]
    if verbose:
        print(f"solver: {F.CYAN}{solver.get_name()}{F.RESET}")

    # Solve
    solutions = solver.solve(
        instance, predictor, verbose, procs, verbose=verbose, samples=samples
    )
    if verbose:
        if len(solutions) == 0:
            print(f"found {F.RED}no solution{F.RESET}")
        else:
            best_cost = min(sol.cost() for sol in solutions)
            print(f"found {F.GREEN}{len(solutions)}{F.RESET} solutions")
            print(
                f"the minimal cost solution found is {F.GREEN}{best_cost}{F.RESET} ({F.GREEN}{best_cost/ len(instance.tests):.1%}{F.RESET})"
            )

    # Save solution
    with open(args.output, "w") as fd:
        json.dump(Solution.to_json(solutions), fd)
