import numpy as np
import pandas as pd


def fixed_point(g, p0, tol=1e-10, max_iter=1000, f=None, p_star=None):
    """Fixed point iteration p_(n+1) = g(p_n); stops when |p_(n+1) - p_n| < tol.

    f      : optional function f(x) = 0, adds the residual f(p)
    p_star : optional exact fixed point, adds error

    Returns a DataFrame, one row per iterate: n, p, g(p), step, [f(p)], f_evals [, error]
    The last row is the final iterate. The table stops early if g leaves the real numbers or
    the values exceed 1e12.
    """
    p = p0
    rows = []

    for n in range(max_iter):
        try:
            p_next = g(p)
            if isinstance(p_next, complex) or abs(p_next) > 1e12:
                break
        except (ValueError, OverflowError, ZeroDivisionError):
            break

        row = [n, p, p_next, abs(p_next - p)]
        if f is not None:
            row.append(f(p))
        rows.append(row + [n + 1])

        p = p_next
        if rows[-1][3] < tol:
            break

    final = [len(rows), p, np.nan, np.nan]
    if f is not None:
        final.append(f(p))
    rows.append(final + [len(rows)])

    columns = ["n", "p", "g(p)", "step"] + (["f(p)"] if f is not None else []) + ["f_evals"]
    table = pd.DataFrame(rows, columns=columns)
    if p_star is not None:
        table["error"] = np.abs(table["p"] - p_star)
    return table
