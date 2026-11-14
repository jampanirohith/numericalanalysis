import numpy as np
import pandas as pd


def sor(A, b, omega, x0=None, tol=1e-6, max_iter=1000):
    """Successive over-relaxation for A x = b; stops when max|x_k - x_(k-1)| < tol.

    omega : relaxation parameter (omega = 1 is Gauss-Seidel)
    x0    : starting vector (zeros by default)

    Returns a DataFrame, one row per iteration (k = 0 is the starting vector):
    k, omega, step, residual, q, x1, x2, ...
    step = max|x_k - x_(k-1)|, residual = max|b - A x_k|, q = residual_k / residual_(k-1).
    The solution is the last row of x1, x2, ...
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    xs = [x]
    step = [np.nan]
    residual = [np.max(np.abs(b - A @ x))]

    for k in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_gs = (b[i] - sum1 - sum2) / A[i, i]
            x_new[i] = (1 - omega) * x[i] + omega * x_gs

        step.append(np.max(np.abs(x_new - x)))
        x = x_new
        xs.append(x)
        residual.append(np.max(np.abs(b - A @ x)))
        if step[-1] < tol:
            break

    table = pd.DataFrame({"k": range(len(xs)), "omega": omega, "step": step, "residual": residual})
    table["q"] = table["residual"] / table["residual"].shift(1)
    return pd.concat([table, pd.DataFrame(xs, columns=[f"x{i + 1}" for i in range(n)])], axis=1)
