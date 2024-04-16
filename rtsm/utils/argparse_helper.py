from typing import Callable


def positive_int(string: str) -> int:
    """
    Return only positive integers and throw errors otherwise.
    """
    i = int(string)
    if i <= 0:
        raise ValueError(f"only accepting positive integer arguments, '{i}' is not!")
    return i


def bounded_float(lower_bound: float, upper_bound: float) -> Callable[[str], bool]:
    """
    Returns a filter that only accepts string that can be parsed into float in the interval [lower_bound; upper_bound]
    """

    def only_bounded_float(string: str) -> bool:
        f = float(string)
        if not (f >= lower_bound and f <= upper_bound):
            raise ValueError(
                f"only accepting float in the interval [{lower_bound}; {upper_bound}], '{f}' is not!"
            )
        return f

    return only_bounded_float
