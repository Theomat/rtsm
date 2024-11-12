"""
Script that enables to sample a sub instance (in the variants space) of an instance.
"""

if __name__ == "__main__":
    import argparse
    import csv

    from rtsm.utils.color_helper import get_color_helper

    F = get_color_helper()

    from rtsm.helper import get_data_loaders, try_load_instance

    # Data Loaders
    data_loaders, supported_extensions = get_data_loaders()

    parser = argparse.ArgumentParser(
        description="Sample a sub instance of an instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        help=f"file containing the data, supported extensions are: {', '.join(supported_extensions)}",
    )
    parser.add_argument("--swap", action="store_true", help="swap variants and tests")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="seed used for sampling the sub instance",
    )
    parser.add_argument(
        "fraction",
        type=float,
        help="fraction of variants to keep",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./sub_instance.csv",
        help="destination CSV file containing the sub instance",
    )
    args = parser.parse_args()
    swap: bool = args.swap
    seed: int = args.seed
    fraction: float = args.fraction
    dst: str = args.output

    assert fraction > 0 and fraction < 1

    # Check then load data
    instance = try_load_instance(args.file, data_loaders, swap)
    print(
        f"loaded {F.CYAN}{len(instance.variants)}{F.RESET} variants and {F.CYAN}{len(instance.tests)}{F.RESET} tests"
    )
    swapped_instance = instance.swap()

    import numpy as np

    np.random.seed(seed)
    selected_variants = np.random.choice(
        swapped_instance.tests,
        size=int(len(swapped_instance.tests) * fraction),
        replace=False,
    )
    sub_swapped_instance = swapped_instance.subset(list(selected_variants))
    sub_instance = sub_swapped_instance.swap()
    # Now we need to save sub_isntance
    with open(dst, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["test", "variant", *sub_instance.performances])
        for i, test in enumerate(sub_instance.tests):
            for j, variant in enumerate(sub_instance.variants):
                perfs = sub_instance.performance_matrix[:, j, i].tolist()
                writer.writerow([test, variant, *perfs])

    test = try_load_instance(dst, data_loaders, swap)
    print(
        f"succesfully generated a sub instance {F.CYAN}{len(test.variants)}{F.RESET} variants and {F.CYAN}{len(test.tests)}{F.RESET} tests"
    )
