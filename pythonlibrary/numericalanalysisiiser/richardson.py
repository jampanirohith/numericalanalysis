import numpy as np
import pandas as pd

from .finite_differences import FORMULAS


def richardson_triangle(first_column, p=2):
    """Richardson table as a list of rows: R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1]) / (2^(p*j) - 1).

    first_column[k] is computed with step h / 2^k.
    p = 2 for central differences and the trapezoidal rule, p = 1 for forward / backward differences.
    """
    R = []
    for k in range(len(first_column)):
        row = [first_column[k]]
        for j in range(1, k + 1):
            row.append(row[j - 1] + (row[j - 1] - R[k - 1][j - 1]) / (2 ** (p * j) - 1))
        R.append(row)
    return R


def richardson(f, x, h, levels=1, method="central", exact=None, layout="by_h"):
    """Richardson extrapolation of a finite difference formula at the point x.

    method : "forward", "backward", "central" or "second"
    exact  : optional true derivative, adds R<j>_error and R<j>_order
    layout : "by_h"     h is a list of steps; R0 = D(h), R1 = (4 D(h/2) - D(h)) / 3, ...
             "triangle" h is one step h0; row k uses h0 / 2^k and R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1]) / (4^j - 1)

    Returns a DataFrame: h, R0, R1, ...  (triangle: k, h, R0, R1, ...)
    """
    D = lambda step: FORMULAS[method](f, x, step)
    p = 1 if method in ("forward", "backward") else 2

    if layout == "by_h":
        h = np.atleast_1d(np.asarray(h, dtype=float))
        rows = []
        for step in h:
            values = [D(step / 2**m) for m in range(levels + 1)]
            row = [values[0]]
            for j in range(1, levels + 1):
                factor = 2 ** (p * j)
                values = [(factor * values[m + 1] - values[m]) / (factor - 1) for m in range(len(values) - 1)]
                row.append(values[0])
            rows.append(row)
        table = pd.DataFrame(rows, columns=[f"R{j}" for j in range(levels + 1)])
        table.insert(0, "h", h)
    elif layout == "triangle":
        steps = [h / 2**k for k in range(levels + 1)]
        table = pd.DataFrame(richardson_triangle([D(step) for step in steps], p))
        table.columns = [f"R{j}" for j in range(levels + 1)]
        table.insert(0, "h", steps)
        table.insert(0, "k", range(levels + 1))
    else:
        raise ValueError("layout must be 'by_h' or 'triangle'")

    if exact is not None:
        for j in range(levels + 1):
            error = np.abs(table[f"R{j}"].to_numpy() - exact)
            with np.errstate(divide="ignore", invalid="ignore"):
                if layout == "by_h":
                    order = np.append(np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:]), np.nan)
                else:
                    order = np.append(np.nan, np.log2(error[:-1] / error[1:]))
            table[f"R{j}_error"] = error
            table[f"R{j}_order"] = order

    return table
