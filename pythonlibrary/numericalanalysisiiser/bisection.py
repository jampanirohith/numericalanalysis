import numpy as np
import pandas as pd


def bisection(f, a, b, tol=1e-10, max_iter=1000, p_star=None):
    """Bisection method for f(x) = 0 on [a, b]; stops when (b - a) / 2 < tol.

    p_star : optional exact root, adds error

    Returns a DataFrame, one row per iteration: n, a, b, p, f(p), b-a, bound, f_evals [, error]
    bound = (b0 - a0) / 2^(n+1). The root is the last p.
    """
    fa = f(a)
    if fa * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    a0, b0 = a, b
    evals = 2
    rows = []

    for n in range(max_iter):
        p = (a + b) / 2
        fp = f(p)
        evals += 1
        rows.append([n, a, b, p, fp, b - a, (b0 - a0) / 2 ** (n + 1), evals])

        if fp == 0 or (b - a) / 2 < tol:
            break
        if fa * fp < 0:
            b = p
        else:
            a = p
            fa = fp

    table = pd.DataFrame(rows, columns=["n", "a", "b", "p", "f(p)", "b-a", "bound", "f_evals"])
    if p_star is not None:
        table["error"] = np.abs(table["p"] - p_star)
    return table
