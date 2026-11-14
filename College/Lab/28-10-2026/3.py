def jacobi_matrix_method(n=55, tol=1e-8, max_iter=500):
    A = [[0.0] * n for _ in range(n)]

    for i in range(n):
        A[i][i] = 4.0
        if i>0:
            A[i][i-1]=-1.0
        if i < n-1:
            A[i][i+1]=-1.0
    
    b=[1.0]*n
    x=[0.0]*n

    residual = []
    num_interations = 0

    for k in range(1, max_iter+1):
        x_new = [0.0] * n
        for i in range(n):
            sum_off_diag = 0.0
            for j in range(n):
                if i != j:
                    sum_off_diag += A[i][j] * x[j]
            x_new[i] = (b[i] - sum_off_diag) / A[i][i]
        
        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        
        res_vector =  [
            abs(b[i] - sum(A[i][j] * x_new[j] for j in range(n))) for i in range(n)
        ]

        res_norm = max(res_vector)
        residual.append(res_norm)

        x = x_new
        num_interations = k

        if diff < tol:
            break
    return x, num_interations, residual

solution, num_iterations, residual = jacobi_matrix_method()

print("Solution:", solution)
print("\nNumber of iterations:", num_iterations)
print("\nResidual:", residual)