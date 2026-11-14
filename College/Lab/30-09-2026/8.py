import numpy as np

def f(x):
    return np.exp(2*x)*np.sin(3*x)

def F(x):
    return np.exp(2*x)*np.cos(3*x)*3 + 2*np.exp(2*x)*np.sin(3*x)

def compositetrapezodial(a,b):
    comptrap = []

    n = [4,8,16,32,64]
    h=[]
    for i in range(0,4):
        h.append((b-a)/n[i])

    for m in range (0,4):
        for j in range(1,n[m]-1):
            comptrap[m] = h[m]/2*(f(a)+f(b)+2*f(a+j*h))
    return comptrap

def compositesimpsons(h,a,b):
    for k in range(1,n):
        while k%2==0:
            even += 2*f(a+k*h)
        while k%2!=0:
            odd += 4*f(a+k*h)
    return h/3(f(a)+f(b)+even+odd)

a = 0
b = 2

exact = F(b) - F(a)


n = [4,8,16,32,64]
h=[]
for i in range(0,4):
    h.append((b-a)/n[i])


trapezod = compositetrapezodial(a,b)
err_trap = abs(exact - trapezod)

simp = compositesimpsons(h, a,b)
err_simp = abs(exact - simp)

print("Exact Value:", exact)
print("Trapezoidal:", trapezod, "Error:",err_trap)
print("Simpson's 1/3:",  simp, "Error:", err_simp)






h2 = (b-a)/2
print(compositetrapezodial(h1,a,b))
print(simpsons(h2,a,b))