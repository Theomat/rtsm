import sys


class AlwaysEmptyString:
    def __getattribute__(self, name: str) -> str:
        return ""


def get_color_helper():
    """
    Returns a seamless interface so that colors are enabled for tty but disabled otherwise.
    """
    if sys.stdout.isatty():
        from colorama import Fore as F

        return F
    else:
        return AlwaysEmptyString()
