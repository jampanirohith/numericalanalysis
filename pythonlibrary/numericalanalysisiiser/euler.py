import numpy as np
import pandas as pd


def euler(f, t0, T, y0, h, exact=None):
    """Explicit Euler method for y' = f(t, y), y(t0) = y0 on [t0, T].

    h     : step size, or a list of step sizes
    exact : optional exact solution y(t), adds exact and error

    Returns a DataFrame (the step sizes stacked): h, t, y, stable [, exact, error]
    stable is False when |y| exceeds 1e10; y is then nan from that step on.
    """
    tables = []

    for step in np.atleast_1d(h):
        N = int(round((T - t0) / step))
        if abs(N * step - (T - t0)) > 1e-9:
            raise ValueError("T - t0 must be a multiple of h")

        t = t0 + step * np.arange(N + 1)
        y = np.full(N + 1, np.nan)
        y[0] = y0
        stable = True

        for i in range(N):
            y[i + 1] = y[i] + step * f(t[i], y[i])
            if not abs(y[i + 1]) < 1e10:
                stable = False
                y[i + 1:] = np.nan
                break

        table = pd.DataFrame({"h": step, "t": t, "y": y, "stable": stable})
        if exact is not None:
            table["exact"] = [exact(ti) for ti in t]
            table["error"] = (table["exact"] - table["y"]).abs()
        tables.append(table)

    return pd.concat(tables, ignore_index=True)
