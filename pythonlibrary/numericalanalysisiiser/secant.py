import numpy as np
import pandas as pd


def secant(f, p0, p1, tol=1e-12, max_iter=100, p_star=None):
    """Secant method; stops when |p_(n+1) - p_n| < tol.

    p_star : optional exact root, adds error and q

    Returns a DataFrame, one row per iterate: n, p_prev, p, f(p), p_next, step, f_evals [, error, q]
    The last row is the final iterate. q = log(e_(n+1)/e_n) / log(e_n/e_(n-1)) is the estimated order.
    """
    p_prev, p_curr = p0, p1
    f_prev, f_curr = f(p0), f(p1)
    evals = 2
    rows = []

    for n in range(1, max_iter + 1):
        if f_curr == f_prev:
            break
        p_next = p_curr - f_curr * (p_curr - p_prev) / (f_curr - f_prev)
        f_next = f(p_next)
        evals += 1
        rows.append([n, p_prev, p_curr, f_curr, p_next, abs(p_next - p_curr), evals])

        p_prev, f_prev = p_curr, f_curr
        p_curr, f_curr = p_next, f_next
        if rows[-1][5] < tol:
            break

    rows.append([len(rows) + 1, p_prev, p_curr, f_curr, np.nan, np.nan, evals])
    table = pd.DataFrame(rows, columns=["n", "p_prev", "p", "f(p)", "p_next", "step", "f_evals"])

    if p_star is not None:
        e_prev, e_curr, e_next = (np.abs(table[c].to_numpy() - p_star) for c in ("p_prev", "p", "p_next"))
        mask = np.minimum(np.minimum(e_prev, e_curr), e_next) < 1e-15          # round-off sized errors are ignored
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.log(e_next / e_curr) / np.log(e_curr / e_prev)
        q[mask] = np.nan
        table["error"] = e_curr
        table["q"] = q
    return table
