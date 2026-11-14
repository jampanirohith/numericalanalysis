import pandas as pd

from .richardson import richardson_triangle
from .trapezoid import composite_trapezoid


def romberg(f, a, b, levels=6, tol=None, exact=None):
    """Romberg integration of f over [a, b].

    levels : number of rows, row k uses n = 2^k sub-intervals
    tol    : if given, rows are added until |R[k][k] - R[k-1][k-1]| < tol
    exact  : optional true value, adds the error of the diagonal entries

    Returns a DataFrame: n, R0, R1, ... [, error]. The last entry of the last row is the answer.
    """
    T = []
    while True:
        T.append(composite_trapezoid(f, a, b, 2 ** len(T)))
        R = richardson_triangle(T)
        if tol is None and len(T) >= levels:
            break
        if tol is not None and (len(T) >= 25 or (len(T) >= 2 and abs(R[-1][-1] - R[-2][-1]) < tol)):
            break

    table = pd.DataFrame(R)
    table.columns = [f"R{j}" for j in range(len(R))]
    table.insert(0, "n", [2**k for k in range(len(R))])
    if exact is not None:
        table["error"] = [abs(R[k][k] - exact) for k in range(len(R))]
    return table
