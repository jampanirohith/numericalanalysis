import math
import matplotlib.pyplot as plt

def bisection(f,a,b,max_iter=1000, tol=1e-10):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None

    c = []
    an = []
    bn = []
    for i in range(max_iter):
        c.append((a + b) / 2)
        if abs(f(c[i])) < tol or (b - a) / 2 < tol:
            break
        if f(c[i]) * f(a) < 0:
            b = c[i]
        else:
            a = c[i]

        an.append(a)
        bn.append(b)

    print("n | a | b | p  | f(p) | b-a")
    print("-----------------------------")
    for n in range(i):
        print(f"{n} | {an[n]} | {bn[n]} | {c[n]} | {f(c[n])} | {bn[n]} - {an[n]}")

    return c[-1]

def f(x):
    return x**3 - x - 2

print(bisection(f, 1, 2))


plt.loglog()
plt.show()





