import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 1/(1+25*(x**2))

N_values = [10,20,50]
x_fine = np.linspace(-1, 1, 1000)
f_fine = f(x_fine)

plt.figure(figsize=(12,8))


print("        f(x) | Maximum Error")
print("        --------------------")


for idx, N in enumerate(N_values, 1):
    x = np.linspace(-1, 1, N+1)
    y = f(x)
    h = np.diff(x)

    A = np.zeros((N+1, N+1))
    rhs = np.zeros(N+1)
    
    A[0,0] = 1
    A[N,N] = 1

    for i in range(1,N):
        A[i, i-1] = h[i-1]/6
        A[i,i] = (h[i-1]+h[i])/3
        A[i, i+1] = h[i]/6
        rhs[i] = (y[i+1] - y[i])/h[i] - (y[i]-y[i-1])/h[i-1]

    M=np.linalg.solve(A, rhs)

    S_fine = np.zeros_like(x_fine)
    for k, x_val in enumerate(x_fine):
        i = np.searchsorted(x, x_val)-1
        i = np.clip(i, 0, N-1)

        dx_right = x[i+1] - x_val
        dx_left = x_val - x[i]

        S_fine[k] = (
            M[i]*(dx_right**3)/(6*h[i]) + 
            M[i+1]*(dx_left**3)/(6*h[i]) + 
            (y[i] - M[i]*(h[i]**2)/6)*(dx_right/h[i]) +
            (y[i]+1 - M[i]+1*(h[i]**2)/6)*(dx_left/h[i])
        )
    
    max_error = np.max(np.abs(f_fine - S_fine))

    print(f"{N:10}   | {max_error:<20.8e}")

    plt.subplot(2,2,idx)
    plt.plot(x_fine, f_fine)
    plt.plot(x_fine, S_fine)
    plt.plot(x,y)

plt.show()