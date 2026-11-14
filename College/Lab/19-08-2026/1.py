import matplotlib.pyplot as plt
import numpy as np

def runge(x):
    return 1/(1+25*(x**2))

def df(x):
    return (-50*x)/((1+25*(x**2))**2)

def hermite(x, runge, df, P):
    n = len(x)

    a = np.repeat(x, 2)
    M = np.zeros((2*n, 2*n))

    M[:, 0] = np.repeat(runge, 2)

    for i in range(n):
        if a[i] == a[i-1]:
              M[i, 1] = df[i//2]
        else:M[i, 1] =(M[i, 0] - M[i-1, 0])/(a[i] - a[i-1])

    for j in range(2, 2*n):
        for i in range(j, 2*n):
             M[i, j] = (M[i, j-1] - M[i-1, j-1]) /(a[i]-a[i-j])

    c = np.diag(M)
    r = np.zeros_like(P)

    for k in range(2*n-1, -1, -1):
        r = c[k] + (P - a[k])*r
    
    return r


N = [5, 10, 20, 50, 300]
P = np.linspace(-1, 1, 5000)


for N in N:
    x = np.linspace(-1, 1, N)
    f = runge(x)
    dF = df(x)
    y = hermite(x, f, dF, P)


plt.plot(P, runge(P))
plt.plot(P, y)
plt.scatter(x, f)

plt.show()