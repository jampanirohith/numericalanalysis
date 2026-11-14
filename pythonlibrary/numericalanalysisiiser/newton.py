import numpy as np
import pandas as pd


def newton(f, df, p0, tol=1e-12, max_iter=100, p_star=None):
    """Newton's method; stops when |p_(n+1) - p_n| < tol.

    df     : derivative f'(x)
    p_star : optional exact root, adds error and q

    Returns a DataFrame, one row per iterate: n, p, f(p), f'(p), step, f_evals, df_evals [, error, q]
    The last row is the final iterate. q = log(e_(n+1)/e_n) / log(e_n/e_(n-1)) is the estimated order.
    """
    p = p0
    rows = []

    for n in range(max_iter):
        fp, dfp = f(p), df(p)
        p_next = p - fp / dfp
        rows.append([n, p, fp, dfp, abs(p_next - p), n + 1, n + 1])
        p = p_next
        if rows[-1][4] < tol:
            break

    rows.append([len(rows), p, f(p), np.nan, np.nan, len(rows), len(rows)])
    table = pd.DataFrame(rows, columns=["n", "p", "f(p)", "f'(p)", "step", "f_evals", "df_evals"])

    if p_star is not None:
        error = np.abs(table["p"].to_numpy() - p_star)
        e = np.where(error < 1e-15, np.nan, error)        # round-off sized errors are ignored
        q = np.full(len(e), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            q[1:-1] = np.log(e[2:] / e[1:-1]) / np.log(e[1:-1] / e[:-2])
        table["error"] = error
        table["q"] = q
    return table
