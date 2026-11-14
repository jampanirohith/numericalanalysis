import numpy as np
import pandas as pd


def gauss_legendre(f, a, b, n=2, exact=None):
    """Gauss-Legendre quadrature for the integral of f over [a, b].

    n     : number of Gauss points, or a list of them
    exact : optional true value, adds error

    Returns a DataFrame: n, value [, error]
    """
    n = np.atleast_1d(n)
    values = []

    for k in n:
        points, weights = np.polynomial.legendre.leggauss(int(k))
        total = 0
        for i in range(k):
            x = (a + b) / 2 + ((b - a) / 2) * points[i]
            total += weights[i] * f(x)
        values.append(((b - a) / 2) * total)

    table = pd.DataFrame({"n": n, "value": values})
    if exact is not None:
        table["error"] = np.abs(table["value"].to_numpy() - exact)
    return table
