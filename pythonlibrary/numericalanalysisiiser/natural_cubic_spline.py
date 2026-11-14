import numpy as np
import pandas as pd


def natural_cubic_spline(x, y, alpha, exact=None):
    """Natural cubic spline through the points (x, y), evaluated at alpha.

    alpha : a number or a list of points
    exact : optional function f(x), adds the columns exact and error

    Returns a DataFrame: alpha, p [, exact, error]
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[0, 0] = 1.0
    A[n, n] = 1.0

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    M = np.linalg.solve(A, b)

    p = np.zeros(len(alpha))
    for idx, t in enumerate(alpha):
        i = int(np.clip(np.searchsorted(x, t) - 1, 0, n - 1))
        dx_right = x[i + 1] - t
        dx_left = t - x[i]
        p[idx] = (
            M[i] * dx_right**3 / (6.0 * h[i])
            + M[i + 1] * dx_left**3 / (6.0 * h[i])
            + (y[i] - M[i] * h[i]**2 / 6.0) * dx_right / h[i]
            + (y[i + 1] - M[i + 1] * h[i]**2 / 6.0) * dx_left / h[i]
        )

    table = pd.DataFrame({"alpha": alpha, "p": p})
    if exact is not None:
        table["exact"] = [exact(a) for a in alpha]
        table["error"] = (table["exact"] - table["p"]).abs()
    return table
