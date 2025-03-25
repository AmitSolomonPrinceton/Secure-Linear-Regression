import numpy as np


def round(x: np.ndarray) -> np.ndarray:
    """
    This function rounds the elements of x.
    values whose decimal part is .5 are rounded up (instead of down as is Numpy's default).
    """
    return np.floor(x + 0.5).astype(int)