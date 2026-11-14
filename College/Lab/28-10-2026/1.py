def jacobi_method(A, b, x0, tol=1e-6, max_iterations=100):
    n = len(b)
    x = x0
    for k in range(max_iterations):
        x_new = [0.0] * n
        for i in range(n):
            sum_off_diag = 0.0
            for j in range(n):
                if i != j:
                    sum_off_diag += A[i][j] * x[j]
            x_new[i] = (b[i] - sum_off_diag) / A[i][i]

        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        x = x_new
        if diff < tol:
            break
    return x

A = [
    [10.0, -1.0, 2.0, 0.0],
    [-1.0, 11.0, -1.0, 3.0],
    [2.0, -1.0, 10.0, -1.0],
    [0.0, 3.0, -1.0, 8.0]
]

b = [6.0, 25.0, -11.0, 15.0]
x0 = [0.0, 0.0, 0.0, 0.0]

solution, num_iterations = jacobi_method(A, b, x0)
r1 = abs(b[0] - A[0][0]*solution[0] + A[0][1]*solution[1] + A[0][2]*solution[2] + A[0][3]*solution[3])
r2 = abs(b[1] - A[1][0]*solution[0] + A[1][1]*solution[1] + A[1][2]*solution[2] + A[1][3]*solution[3])
r3 = abs(b[2] - A[2][0]*solution[0] + A[2][1]*solution[1] + A[2][2]*solution[2] + A[2][3]*solution[3])
r4 = abs(b[3] - A[3][0]*solution[0] + A[3][1]*solution[1] + A[3][2]*solution[2] + A[3][3]*solution[3])

residual_norm = max(r1,r2,r3,r4)

print("Solution:", solution)
print("Number of iterations:", num_iterations)
print("Residual norm:", residual_norm)