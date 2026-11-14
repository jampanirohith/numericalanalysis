import math

def fixedpoint(e, x0, max_iter=1000, tol=1e-10):
    k = 0
    xi = [x0]
    for i in range(max_iter):
        xi.append(e(x0))
        if abs(xi[i] - x0) < tol:
            break
        
        k = k+1
        return xi[i]

    print(f"Fixed point iteration using: {e.__name__}")
    print("Maximum number of interations:", k)



def f(x):
    return x**3 + x - 1

def g1(x):
    return (1 - x)**(1/3)

def g2(x):
    return 1-x**3

def g3(x):
    return (1+x)/3

print(fixedpoint(g1, 0.5, 1000, 1e-10))



