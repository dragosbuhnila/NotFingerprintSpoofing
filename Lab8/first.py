import numpy as np
from scipy.optimize import fmin_l_bfgs_b  # type: ignore
from typing import Tuple


# Implements f(y, z) = (y + 3)2 + sin(y) + (z + 1)2
def f_with_gradient(x: np.ndarray) -> Tuple[float, np.ndarray]:
    if x.shape != (2,):
        raise ValueError('x must be a 2D array with shape (2,)')
    f_ret = f(x)

    gradient = np.array([2 * (x[0] + 3) + np.cos(x[0]), 2 * (x[1] + 1)])

    return f_ret, gradient


def f(x: np.ndarray) -> float:
    if x.shape != (2,):
        raise ValueError('x must be a 2D array with shape (2,)')
    f_ret = (x[0] + 3) ** 2 + np.sin(x[0]) + (x[1] + 1) ** 2

    return f_ret


if __name__ == '__main__':
    x0 = np.array([0, 0])

    x, y, d = fmin_l_bfgs_b(f_with_gradient, x0)

    print(f"With provided graident: x is {x}, f(x) is {y}. d is {d}.")

    x, y, d = fmin_l_bfgs_b(f, x0, approx_grad=True)

    print(f"With approximated graident: x is {x}, f(x) is {y}. d is {d}.")