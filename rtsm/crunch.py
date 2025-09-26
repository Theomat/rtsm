if __name__ == "__main__":
    import argparse
    import sys
    import os
    import json
    import atexit
    import time

    from rtsm.instance import Instance
    from rtsm.solution import Solution
    from rtsm.predictors.predictor import Predictor
    from rtsm.solvers.fusion_solver import FusionSolver
    from rtsm.helper import (
        get_predictors,
        get_data_loaders,
        get_solvers,
        try_load_instance,
    )

    from rtsm.utils.argparse_helper import positive_int, bounded_float
    from rtsm.utils.color_helper import get_color_helper

    F = get_color_helper()

    data_loaders, supported_extensions = get_data_loaders()
    predictors = get_predictors()
    solvers = get_solvers()

    parser = argparse.ArgumentParser(
        description="Crunch ranked test suite minimisation",
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
    group.add_argument(
        "--start",
        type=str,
        help="start from an existing solution file in order to improve upon it",
    )

    group = parser.add_argument_group("sampling based solvers (rs, bisect)")
    group.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed used for probabilistic solvers",
    )
    group.add_argument(
        "--tries",
        type=int,
        default=1,
        help="number of attempts tried with no improvement before giving up",
    )

    group = parser.add_argument_group("approximate (weighted, linear, logistic-rank)")
    group.add_argument(
        "--kendall",
        type=bounded_float(0, 1),
        default=1.0,
        help="Kendall coefficient of the ranking needed",
    )

    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--no-autosave", action="store_true")

    parser.add_argument(
        "-p",
        "--procs",
        type=positive_int,
        default=1,
        help="number of processors to use",
    )
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
    no_autosave: bool = args.no_autosave
    procs: int = args.procs
    tries: int = args.tries

    initial_solution: str = args.start or ""
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

    # Warm start with previous solution
    if len(initial_solution) > 0:
        if not os.path.exists(initial_solution) or not os.path.isfile(initial_solution):
            print(
                f"{F.RED}start solution file does not exist or is not a valid file:{F.RESET}",
                initial_solution,
                file=sys.stderr,
            )
            sys.exit(1)
        with open(initial_solution) as fd:
            solution_list = json.load(fd)
            if len(solution_list) == 0:
                print(
                    f"{F.RED}start solution file contains no solution!{F.RESET}",
                    file=sys.stderr,
                )
                sys.exit(1)
            one_sol = solution_list[0]
            instance.set_start(one_sol)
            if verbose:
                print(
                    f"used start solution to go from {F.CYAN}{len(instance.tests)}{F.RESET} tests to {F.CYAN}{len(one_sol)}{F.RESET} ({F.CYAN}{len(one_sol) / len(instance.tests):.1%}{F.RESET}) tests"
                )

    if verbose:
        print(
            f"loaded {F.CYAN}{len(instance.variants)}{F.RESET} variants and {F.CYAN}{len(instance.tests)}{F.RESET} tests"
        )
        if not instance.check_filled():
            print(
                f"{F.YELLOW}warning:{F.RESET} the performance matrix is not completely filled!"
            )

    # Build predictor
    accuracy: float = args.kendall

    def predictor_builder(instance: Instance) -> Predictor:
        return predictors[args.predictor](instance, accuracy=accuracy)

    if verbose:
        print(
            f"prediction model: {F.CYAN}{predictor_builder(instance).get_name()}{F.RESET}"
        )

    # Get solver
    base_solver = solvers[args.solver]

    if verbose:
        print(f"base solver: {F.CYAN}{base_solver.get_name()}{F.RESET}")

    start = time.perf_counter_ns()

    def save(sols, **kwargs):
        end = time.perf_counter_ns() - start
        with open(args.output, "w") as fd:
            json.dump(
                {
                    "solutions": Solution.to_json(sols),
                    "runtime": int(end / 1e6) / 1e3,
                    "base_solver": args.solver,
                    "predictor": args.predictor,
                    "accuracy": accuracy,
                    **kwargs,
                },
                fd,
            )

    solver = FusionSolver(base_solver.__class__, -1)

    # Anytime solving
    def save_result_pre_emptively():
        sols = solver.early_exit()
        if verbose:
            print(
                f"{F.YELLOW}warning:{F.RESET} early stopping, results were still saved!"
            )
        save(sols, improved=improved)

    if not no_autosave:
        atexit.register(save_result_pre_emptively)

    # Solve
    current_cost = Solution(
        instance, [1 for _ in instance.tests], instance.warm_start()
    ).cost()
    size = 20
    solutions = set()
    i = 0
    last_with_progress = i
    improved = 0
    while i - last_with_progress < tries:
        solver.splits = sum(instance.warm_start()) // size
        solutions = solver.solve(
            instance,
            predictor_builder,
            verbose,
            procs,
            verbose=False,
            seed=args.seed + i,
            on_progress_callback=save if not no_autosave else lambda *args: len(args),
        )
        i += 1
        if len(solutions) == 0:
            break
        sols = sorted([(x.cost(), x) for x in solutions], key=lambda x: x[0])
        one_sol = sols[0][1]
        new_cost = one_sol.cost()
        if new_cost < current_cost:
            if not no_autosave:
                save(solutions, improved=improved)
            last_with_progress = i
            improved += 1
            instance.set_start(one_sol.tests)
            current_cost = new_cost
            if len(one_sol.tests) // size <= 1:
                if len(one_sol.tests) == 1:
                    break
                solver = base_solver
    if not no_autosave:
        atexit.unregister(save_result_pre_emptively)
    if verbose:
        if len(solutions) == 0:
            print(f"found {F.RED}no solution{F.RESET}")
        else:
            best_cost = min(sol.cost() for sol in solutions)
            print(
                f"found {F.GREEN}{len(solutions)}{F.RESET} solution{'s' if len(solutions) > 1 else ''}"
            )
            print(
                f"the minimal cost solution found is {F.GREEN}{best_cost}{F.RESET} ({F.GREEN}{best_cost / instance.total_cost():.1%}{F.RESET})"
            )
    # Save solution
    save(solutions, improved=improved)
    if verbose:
        print(f"saved to {F.GREEN}{args.output}{F.RESET}")
