from typing import Optional
from colorama import Fore as F

import tqdm


class ProgressBar:
    """
    Wrapper around tqdm progress bar that automatically color fields and offer the same API whether the progress bar is enabled or not.
    """

    def __init__(self, total: int, name: str, use_tqdm: bool) -> None:
        self.has_bar = use_tqdm
        if use_tqdm:
            self.pbar = tqdm.tqdm(
                total=total, desc=f"{F.LIGHTYELLOW_EX}{name}{F.RESET}", smoothing=1
            )

    def update(self, n: int):
        if self.has_bar:
            self.pbar.update(n)

    def set_best(self, score: int, ratio: Optional[float] = None):
        if self.has_bar:
            additional = (
                "" if ratio is None else f"({F.LIGHTYELLOW_EX}{ratio:.1%}{F.RESET})"
            )
            self.pbar.set_postfix_str(
                f"best: {F.LIGHTYELLOW_EX}{score}{F.RESET} {additional}"
            )

    def close(self):
        if self.has_bar:
            self.pbar.close()
