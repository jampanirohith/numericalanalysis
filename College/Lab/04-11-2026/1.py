import numpy as np
import pandas as pd

def sor(A, b, omega, tol=1e-3, max_iter=1000):
    n = len(b)
    x = np.zeros(n)

    for iteration in range(1, max_iter + 1):
        x_old = np.copy(x)
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            
            x_gs = (b[i] - sum1 - sum2) / A[i, i]
            
            x[i] = (1 - omega) * x_old[i] + omega * x_gs
            
        error = np.max(np.abs(x - x_old))
        
        if error < tol:
            return x, iteration, error

    return x, max_iter, error

A = np.array([
    [ 4.0,  1.0, -1.0,  1.0],
    [ 1.0,  4.0, -1.0, -1.0],
    [-1.0, -1.0,  5.0,  1.0],
    [ 1.0, -1.0,  1.0,  3.0]
], dtype=float)

b = np.array([-2.0, -1.0, 0.0, 1.0], dtype=float)

omega_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9]

results = []

for omega in omega_values:
    sol, iters, final_err = sor(A, b, omega, tol=1e-3)
    
    results.append({
        "Omega (ω)": omega,
        "Iterations": iters,
        "Final l_inf Norm": round(final_err, 6),
        "Solution Vector": np.round(sol, 4).tolist()
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))