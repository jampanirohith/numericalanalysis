import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.exp(-x) * np.sin(2 * x)

def naturalspline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    M = np.linalg.solve(A, b)
    return M

def evaluate_spline(x_nodes, y_nodes, M, x_eval):
    x_eval = np.atleast_1d(x_eval)
    S = np.zeros_like(x_eval)

    for idx, x in enumerate(x_eval):
        i = np.searchsorted(x_nodes, x) - 1
        if i < 0:
            i = 0
        if i >= len(x_nodes) - 1:
            i = len(x_nodes) - 2

        h = x_nodes[i + 1] - x_nodes[i]

        term1 = M[i] * (x_nodes[i + 1] - x) ** 3 / (6 * h)
        term2 = M[i + 1] * (x - x_nodes[i]) ** 3 / (6 * h)
        term3 = (y_nodes[i] - (M[i] * h**2) / 6) * (x_nodes[i + 1] - x) / h
        term4 = (y_nodes[i + 1] - (M[i + 1] * h**2) / 6) * (x - x_nodes[i]) / h

        S[idx] = term1 + term2 + term3 + term4

    return S if len(S) > 1 else S[0]

x_fine = np.linspace(0, 4, 2001)
y_exact = f(x_fine)

mesh_sizes = [0.5, 0.25, 0.125]
errors = []

plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_exact)

for h in mesh_sizes:
    x_nodes = np.arange(0, 4 + h / 2, h)
    y_nodes = f(x_nodes)

    M = naturalspline(x_nodes, y_nodes)
    y_spline = evaluate_spline(x_nodes, y_nodes, M, x_fine)

    E_h = np.max(np.abs(y_exact - y_spline))
    errors.append(E_h)

    plt.plot(x_fine, y_spline)


print(f"{'h':<10}{'E_h':<20}{'E_h / E_{h/2}':<15}")
print("-" * 45)


for idx, h in enumerate(mesh_sizes):
    if idx < len(mesh_sizes) - 1:
        ratio_str = f"{errors[idx] / errors[idx+1]:.4f}"
    else:
        ratio_str = "—"
    print(f"{h:<10}{errors[idx]:<20.4e}{ratio_str:<15}")

plt.xlabel()
plt.show()
