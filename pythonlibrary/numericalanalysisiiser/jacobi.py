import numpy as np
import pandas as pd


def jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    """Jacobi iteration for A x = b; stops when max|x_k - x_(k-1)| < tol.

    x0 : starting vector (zeros by default)

    Returns a DataFrame, one row per iteration (k = 0 is the starting vector):
    k, step, residual, q, x1, x2, ...
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
        x_new = np.zeros(n)
        for i in range(n):
            sum_off_diag = 0.0
            for j in range(n):
                if i != j:
                    sum_off_diag += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_off_diag) / A[i, i]

        step.append(np.max(np.abs(x_new - x)))
        x = x_new
        xs.append(x)
        residual.append(np.max(np.abs(b - A @ x)))
        if step[-1] < tol:
            break

    table = pd.DataFrame({"k": range(len(xs)), "step": step, "residual": residual})
    table["q"] = table["residual"] / table["residual"].shift(1)
    return pd.concat([table, pd.DataFrame(xs, columns=[f"x{i + 1}" for i in range(n)])], axis=1)
