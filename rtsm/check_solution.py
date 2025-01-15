if __name__ == "__main__":
    import argparse
    import sys
    import os
    import json
    import time
    from typing import Tuple

    from scipy.stats import spearmanr, permutation_test
    import numpy as np

    from rtsm.solution import Solution

    def spearman_data(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        rs = spearmanr(x, y)
        dof = len(x) - 2
        if x.shape[0] < 500:

            def statistic(z):
                rz = spearmanr(z, y).statistic
                if abs(rz) >= 1:
                    return rz
                transformed = rz * np.sqrt(dof / ((rz + 1.0) * (1.0 - rz)))
                return transformed

            ref = permutation_test(
                (x,),
                statistic,
                alternative="greater",
                permutation_type="pairings",
                random_state=1,
            )
            return rs.statistic, ref.pvalue
        else:
            return rs.statistic, rs.pvalue

    import numpy as np
    from rtsm.utils.color_helper import get_color_helper

    F = get_color_helper()
    from rtsm.predictors.predictor import Predictor, to_ranking, ranking_error, to_ranks

    from rtsm.helper import get_predictors, get_data_loaders, try_load_instance

    data_loaders, supported_extensions = get_data_loaders()
    predictors = get_predictors()

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
    parser.add_argument(
        "--full",
        type=str,
        help=f"file containing a super instance of the data, supported extensions are: {', '.join(supported_extensions)}",
    )
    parser.add_argument("--swap", action="store_true", help="swap variants and tests")
    # parser.add_argument(
    #     "--predictor",
    #     choices=list(predictors.keys()),
    #     default=list(predictors.keys())[0],
    #     help="prediction model to use",
    # )

    args = parser.parse_args()
    swap: bool = args.swap
    initial_solution: str = args.solution or ""

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
    print(f"target accuracy: {F.CYAN}{data['accuracy']:.2%}{F.RESET}")

    # Measure performance gains
    print("performance gains:")
    my_sol = Solution(instance, tuple(one_sol))
    for perf, (val, total) in my_sol.measure_performances().items():
        print(
            f"\t{perf}: {F.GREEN}{val:.2}{F.RESET} / {F.CYAN}{total:.2}{F.RESET} ({F.GREEN}{val/total:.2%}{F.RESET})"
        )
    print(
        "cost:",
        f"{F.GREEN}{my_sol.cost():.2}{F.RESET} / {F.CYAN}{instance.total_cost():.2}{F.RESET} ({F.GREEN}{my_sol.cost()/instance.total_cost():.2%}{F.RESET})",
    )
    R = to_ranking(np.sum(instance.performance_matrix, axis=-1))
    ranks_original = to_ranks(R)

    n = len(instance.variants)

    sR = to_ranking(np.sum(instance.subset(one_sol).performance_matrix, axis=-1))
    ranks_no_pred = to_ranks(sR)
    print("without prediction:")
    print(f"\tranking error: {F.GREEN}{ranking_error(R, sR):.2%}{F.RESET}")

    def spearman_to_str(rankA, rankB):
        out = []
        for h, perf in zip(range(rankA.shape[0]), instance.performances):
            s, pval = spearman_data(rankA[h], rankB[h])
            out.append(
                f"{perf}:{F.GREEN}{s:.3f}{F.RESET} p-value:{F.GREEN}{pval:.3f}{F.RESET}"
            )
        return " ".join(out)

    spearman_desc = spearman_to_str(ranks_original, ranks_no_pred)
    print(f"\tspearman: {spearman_desc}")
    print("with prediction:")

    start = time.perf_counter()
    mask = [x in one_sol for x in instance.tests]
    ranking_pred = predictor.get_ranking(mask)
    duration = time.perf_counter() - start
    error = ranking_error(R, ranking_pred)
    ranks_pred = to_ranks(ranking_pred)
    print(
        f"\tranking error ({F.LIGHTYELLOW_EX}{duration:.2f}{F.RESET}s): {F.GREEN}{error:.2%}{F.RESET}"
    )
    spearman_desc = spearman_to_str(ranks_original, ranks_pred)
    print(f"\tspearman: {spearman_desc}")

    if args.full is not None:
        print()
        full = try_load_instance(args.full, data_loaders, swap)
        print(
            f"loaded full {F.CYAN}{len(full.variants)}{F.RESET} variants and {F.CYAN}{len(full.tests)}{F.RESET} tests"
        )
        R = to_ranking(np.sum(full.performance_matrix, axis=-1))
        ranks_original = to_ranks(R)

        new_variants = set(full.variants) - set(instance.variants)
        print(
            f"\tnew variants ({F.LIGHTYELLOW_EX}{len(new_variants)}{F.RESET}):",
            ", ".join(new_variants),
        )
        prediction = predictor.export_prediction(mask)
        copy = full.performance_matrix.copy()
        rev_mask = np.logical_not(np.asarray(mask))
        copy[:, :, rev_mask] = 0
        missing = prediction.predict(full.performance_matrix[:, :, mask])
        copy[:, :, rev_mask] = missing

        Rpred = to_ranking(np.sum(copy, axis=-1))
        ranks_pred = to_ranks(Rpred)
        error = ranking_error(R, Rpred)
        print(f"\tranking error: {F.GREEN}{error:.2%}{F.RESET}")
        spearman_desc = spearman_to_str(ranks_original, ranks_pred)
        print(f"\tspearman: {spearman_desc}")
