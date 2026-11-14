from jacobimatrix import jacobimatrix_solution, jacobimatrix_num_iterations, jacobimatrix_residual
from gaussseidelmatrix import gaussseidelmatrix_solution, gaussseidelmatrix_num_iterations, gaussseidelmatrix_residual
import matplotlib.pyplot as plt

print("Jacobi iterations:", jacobimatrix_num_iterations)
print("Gauss-Seidel iterations:", gaussseidelmatrix_num_iterations)

q_j = [jacobimatrix_residual[i] / jacobimatrix_residual[i-1] for i in range(1, len(jacobimatrix_residual))]
q_gs = [gaussseidelmatrix_residual[i] / gaussseidelmatrix_residual[i-1] for i in range(1, len(gaussseidelmatrix_residual))]

print(f"Jacobi asymptotic factor: {q_j[-1]:.5f}")
print(f"Gauss-Seidel asymptotic factor: {q_gs[-1]:.5f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(jacobimatrix_residual) + 1), jacobimatrix_residual)
plt.plot(range(1, len(gaussseidelmatrix_residual) + 1), gaussseidelmatrix_residual)

plt.yscale('log')

plt.show()
