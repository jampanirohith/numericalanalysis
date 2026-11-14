# numericalanalysisiiser

Numerical analysis methods from the MAT 308 Mathematics Lab at IISER Thiruvananthapuram: interpolation,
numerical differentiation and integration, root finding, iterative linear solvers and the Euler method.

Every function returns a **pandas DataFrame**: a table with one row per iteration, step size or point, from
which you can read the answer or plot any columns.

* PyPI: <https://pypi.org/project/numericalanalysisiiser/>
* Source: <https://github.com/jampanirohith/numericalanalysis>

## Install

```bash
pip install numericalanalysisiiser
```

Needs Python 3.9 or newer. numpy and pandas are installed with it. To upgrade: `pip install --upgrade numericalanalysisiiser`.

## How to use a function

1. **Import** the function. After a plain `import numericalanalysisiiser` you would have to write
   `numericalanalysisiiser.lagrange(...)`; importing by name is simpler:

   ```python
   from numericalanalysisiiser import lagrange
   ```

   Other forms that work: `from numericalanalysisiiser import *` (all functions), or
   `import numericalanalysisiiser as na` and then `na.lagrange(...)`.

2. **Call** it with your data. The functions you give it (`f`, `g`, `df`, `exact`) are ordinary Python
   functions, e.g. `lambda x: x**2 - 2` or `math.sin`.

3. **Print** the result. In a `.py` file (also in VS Code) nothing is shown unless you print:

   ```python
   print(lagrange([0, 1, 2, 3], [1, 2, 5, 10], 1.5))
   ```
   ```
      alpha     p
   0    1.5  3.25
   ```

   In a Jupyter notebook or at the Python prompt the table is shown without `print`.

**Wide tables** are shortened by pandas with `...`. Select the columns you want,
`print(t[["n", "p", "error"]])`, or show everything with `print(t.to_string())`.

## Reading the results

* **One value or a list.** Where something can vary (`alpha`, `h`, `n`) give a number or a list; you get one row per value.
* **`exact=` / `p_star=`** is the true value, optional. With it the table gets `error` columns, and `order`
  (observed order of convergence) or `q` (estimated order of an iteration).
* **Iterative methods** (root finding, linear systems) give one row per iteration, and **the last row is the answer.**
* **`order`** in a row compares that row with the next one, so the last row is always `NaN`. List the step sizes from large to small.
* **`NaN`** means "not defined for this row"; it is not an error.

---

# The methods

## Interpolation

Each of the five functions evaluates an interpolating curve at the points `alpha` and returns the columns
`alpha, p` (and `exact, error` if you pass `exact=`, a function).

The examples use the points `(0, 1), (1, 2), (2, 5), (3, 10)`, which lie on `y = x^2 + 1`.

### `lagrange(x, y, alpha, exact=None)`

Lagrange polynomial through the points `(x, y)`. The `x` values must be distinct.

```python
from numericalanalysisiiser import lagrange

print(lagrange([0, 1, 2, 3], [1, 2, 5, 10], [0.5, 1.5], exact=lambda t: t**2 + 1))
```
```
   alpha     p  exact  error
0    0.5  1.25   1.25    0.0
1    1.5  3.25   3.25    0.0
```

The curve is `x^2 + 1`, so the interpolation is exact and the error is zero.

### `newton_interpolation(x, y, alpha, exact=None)`

The same polynomial in Newton's divided-difference form. Same arguments and columns as `lagrange`.

```python
from numericalanalysisiiser import newton_interpolation

print(newton_interpolation([0, 1, 2, 3], [1, 2, 5, 10], [0.5, 1.5]))
```
```
   alpha     p
0    0.5  1.25
1    1.5  3.25
```

### `divided_differences(x, y)`

The divided-difference table. **The first row holds the Newton coefficients.**

```python
from numericalanalysisiiser import divided_differences

print(divided_differences([0, 1, 2, 3], [1, 2, 5, 10]))
```
```
   x  f[0]  f[1]  f[2]  f[3]
0  0     1   1.0   1.0   0.0
1  1     2   3.0   1.0   NaN
2  2     5   5.0   NaN   NaN
3  3    10   NaN   NaN   NaN
```

Here the polynomial is `1 + 1*(x - 0) + 1*(x - 0)(x - 1)`; the last coefficient `0` says the degree is 2.

### `hermite(x, y, dy, alpha, exact=None)`

Hermite interpolation: the polynomial matches the values `y` **and** the slopes `dy` at the nodes.

```python
from numericalanalysisiiser import hermite

# f(x) = x^3, nodes 0 and 1: values 0, 1 and slopes 0, 3
print(hermite([0, 1], [0, 1], [0, 3], [0.25, 0.5], exact=lambda t: t**3))
```
```
   alpha         p     exact  error
0   0.25  0.015625  0.015625    0.0
1   0.50  0.125000  0.125000    0.0
```

A cubic is reproduced exactly from two nodes with their slopes.

### `natural_cubic_spline(x, y, alpha, exact=None)`

Natural cubic spline (second derivative zero at both ends). The `x` values must be increasing.

```python
import numpy as np
from numericalanalysisiiser import natural_cubic_spline

nodes = [0, 1, 2, 3]
print(natural_cubic_spline(nodes, np.sin(nodes), [0.5, 1.5, 2.5], exact=np.sin).round(6))
```
```
   alpha         p     exact     error
0    0.5  0.477200  0.479426  0.002226
1    1.5  0.996108  0.997495  0.001387
2    2.5  0.589468  0.598472  0.009004
```

## Numerical differentiation

### `finite_difference(f, x, h, exact=None, methods=("forward", "backward", "central"))`

Approximates `f'(x)` at the point `x`. `h` is a step size or a list of them. `methods` can contain
`"forward"`, `"backward"`, `"central"` and `"second"` (this one approximates `f''(x)`).

| Method | Formula | Error |
|---|---|---|
| `forward` | `(f(x+h) - f(x)) / h` | order 1 |
| `backward` | `(f(x) - f(x-h)) / h` | order 1 |
| `central` | `(f(x+h) - f(x-h)) / (2h)` | order 2 |
| `second` | `(f(x+h) - 2f(x) + f(x-h)) / h^2` | order 2 |

With `exact=`, each method also gets `<method>_error` and `<method>_order`.

```python
import math
from numericalanalysisiiser import finite_difference

t = finite_difference(math.sin, 1.0, [0.1, 0.05, 0.025], exact=math.cos(1.0))
print(t[["h", "forward", "forward_error", "central", "central_error", "central_order"]].round(6))
```
```
       h   forward  forward_error   central  central_error  central_order
0  0.100  0.497364       0.042939  0.539402       0.000900       1.999459
1  0.050  0.519045       0.021257  0.540077       0.000225       1.999865
2  0.025  0.529728       0.010574  0.540246       0.000056            NaN
```

The central error drops by a factor 4 each time `h` is halved (order 2); the forward error only by 2 (order 1).

### `richardson(f, x, h, levels=1, method="central", exact=None, layout="by_h")`

Richardson extrapolation of a difference formula (`method` as above): it combines the formula at step `h` and
`h/2` to cancel the leading error term. Columns `R0` (the plain formula), `R1`, `R2`, ... get more accurate.

**`layout="by_h"`** (default): `h` is a list; one row per `h`, with `R1 = (4 D(h/2) - D(h)) / 3` for the central formula.

```python
import math
from numericalanalysisiiser import richardson

print(richardson(math.sin, 1.0, [0.2, 0.1, 0.05], exact=math.cos(1.0)).round(8))
```
```
      h        R0        R1  R0_error  R0_order      R1_error  R1_order
0  0.20  0.536707  0.540301  0.003595  1.997836  1.800000e-06  3.998712
1  0.10  0.539402  0.540302  0.000900  1.999459  1.100000e-07  3.999678
2  0.05  0.540077  0.540302  0.000225       NaN  1.000000e-08       NaN
```

`R0` has order 2, `R1` order 4. Use `levels=2` to add `R2` (order 6).

**`layout="triangle"`**: `h` is **one** starting step; row `k` uses `h / 2^k` and the table is the classical Richardson triangle.

```python
print(richardson(math.sin, 1.0, 0.4, levels=3, layout="triangle").round(8))
```
```
   k     h        R0        R1        R2        R3
0  0  0.40  0.526009       NaN       NaN       NaN
1  1  0.20  0.536707  0.540274       NaN       NaN
2  2  0.10  0.539402  0.540301  0.540302       NaN
3  3  0.05  0.540077  0.540302  0.540302  0.540302
```

### `richardson_triangle(first_column, p=2)`

The triangle for values you computed yourself. `first_column[k]` must have been computed with step `h / 2^k`.
Returns a list of rows. Use `p=2` for central differences and the trapezoidal rule, `p=1` for forward / backward.

```python
from numericalanalysisiiser import richardson_triangle

print(richardson_triangle([1.0, 0.75, 0.6875]))
```
```
[[1.0], [0.75, 0.6666666666666666], [0.6875, 0.6666666666666666, 0.6666666666666666]]
```

## Numerical integration

The examples integrate `e^x` over `[0, 1]`, whose exact value is `e - 1 = 1.71828...`.

### `trapezoid(f, a, b, n=1, exact=None)` and `simpson(f, a, b, n=2, exact=None)`

Composite trapezoidal rule and Simpson's 1/3 rule with `n` sub-intervals (`n` is a number or a list; for Simpson it
must be even). `n=1` (trapezoid) and `n=2` (Simpson) are the basic one-application rules. Columns: `n, h, value`
(and `error, order` with `exact=`).

```python
import math
from numericalanalysisiiser import trapezoid, simpson

print(trapezoid(math.exp, 0, 1, [2, 4, 8], exact=math.e - 1).round(6))
print(simpson(math.exp, 0, 1, [2, 4, 8], exact=math.e - 1).round(8))
```
```
   n      h     value     error     order
0  2  0.500  1.753931  0.035649  1.995513
1  4  0.250  1.727222  0.008940  1.998874
2  8  0.125  1.720519  0.002237       NaN
   n      h     value     error     order
0  2  0.500  1.718861  0.000579  3.968247
1  4  0.250  1.718319  0.000037  3.991978
2  8  0.125  1.718284  0.000002       NaN
```

Trapezoid has order 2, Simpson order 4. `composite_trapezoid(f, a, b, n)` and `composite_simpson(f, a, b, n)` return
just the number:

```python
from numericalanalysisiiser import composite_simpson

print(composite_simpson(math.exp, 0, 1, 4))      # 1.718318841921747
```

### `romberg(f, a, b, levels=6, tol=None, exact=None)`

Romberg integration. Column `R0` is the trapezoidal rule with `n = 1, 2, 4, 8, ...`; the other columns are Richardson
extrapolations. **The answer is the last entry of the last row.** With `tol=` rows are added until two successive
diagonal entries differ by less than `tol`. With `exact=` an `error` column (of the diagonal entries) is added.

```python
from numericalanalysisiiser import romberg

print(romberg(math.exp, 0, 1, levels=4, exact=math.e - 1))
```
```
   n        R0        R1        R2        R3         error
0  1  1.859141       NaN       NaN       NaN  1.408591e-01
1  2  1.753931  1.718861       NaN       NaN  5.793234e-04
2  4  1.727222  1.718319  1.718283       NaN  8.594657e-07
3  8  1.720519  1.718284  1.718282  1.718282  3.354852e-10
```

### `gauss_legendre(f, a, b, n=2, exact=None)`

Gauss-Legendre quadrature with `n` points (`n` a number or a list). An `n`-point rule is exact for polynomials up to
degree `2n - 1`. Columns: `n, value` (and `error`).

```python
from numericalanalysisiiser import gauss_legendre

print(gauss_legendre(math.exp, 0, 1, [1, 2, 3], exact=math.e - 1))
```
```
   n     value         error
0  1  1.648721  6.956056e-02
1  2  1.717896  3.854505e-04
2  3  1.718281  8.240865e-07
```

## Root finding

The examples all solve `x^2 - 2 = 0`, i.e. they find `sqrt(2) = 1.41421356...`.

For every method: **the last row is the final iterate, so the root is `t["p"].iloc[-1]`.** `p_star` is the exact
root; give it to get the `error` column (and `q`, the estimated order of convergence). `f_evals` / `df_evals` count the
evaluations of `f` / `f'`. The number of iterations is `len(t)` for `bisection` and `len(t) - 1` for the others.

### `bisection(f, a, b, tol=1e-10, max_iter=1000, p_star=None)`

Needs `f(a)` and `f(b)` of opposite sign (otherwise `ValueError`). Stops when `(b - a) / 2 < tol`.
Columns: `n, a, b, p, f(p), b-a, bound, f_evals` (+ `error`). `bound = (b0 - a0) / 2^(n+1)` is the guaranteed error bound.

```python
import math
from numericalanalysisiiser import bisection

f = lambda x: x**2 - 2
t = bisection(f, 1, 2, tol=1e-3, p_star=math.sqrt(2))
print(t[["n", "p", "f(p)", "bound", "error"]].round(6))
```
```
   n         p      f(p)     bound     error
0  0  1.500000  0.250000  0.500000  0.085786
1  1  1.250000 -0.437500  0.250000  0.164214
2  2  1.375000 -0.109375  0.125000  0.039214
3  3  1.437500  0.066406  0.062500  0.023286
4  4  1.406250 -0.022461  0.031250  0.007964
5  5  1.421875  0.021729  0.015625  0.007661
6  6  1.414062 -0.000427  0.007812  0.000151
7  7  1.417969  0.010635  0.003906  0.003755
8  8  1.416016  0.005100  0.001953  0.001802
9  9  1.415039  0.002336  0.000977  0.000826
```

The error always stays below `bound`. With the default `tol=1e-10` it takes 34 iterations
(`math.ceil(math.log2((b - a) / tol))`).

### `fixed_point(g, p0, tol=1e-10, max_iter=1000, f=None, p_star=None)`

Iterates `p_(n+1) = g(p_n)` until `abs(p_(n+1) - p_n) < tol`. It converges to a fixed point `p` of `g` when
`abs(g'(p)) < 1`. Give `f` to also get the residual column `f(p)`; `p_star` is the exact *fixed point* of `g`.
Columns: `n, p, g(p), step, [f(p)], f_evals` (+ `error`). If `g` fails (error, complex number, values above `1e12`) the
table simply stops there.

```python
from numericalanalysisiiser import fixed_point

# x = (x + 2/x) / 2 has the fixed point sqrt(2)
t = fixed_point(lambda x: (x + 2 / x) / 2, 1.0, tol=1e-10, f=f, p_star=math.sqrt(2))
print(t)
```
```
   n         p      g(p)          step          f(p)  f_evals         error
0  0  1.000000  1.500000  5.000000e-01 -1.000000e+00        1  4.142136e-01
1  1  1.500000  1.416667  8.333333e-02  2.500000e-01        2  8.578644e-02
2  2  1.416667  1.414216  2.450980e-03  6.944444e-03        3  2.453104e-03
3  3  1.414216  1.414214  2.123900e-06  6.007305e-06        4  2.123901e-06
4  4  1.414214  1.414214  1.594946e-12  4.510614e-12        5  1.594724e-12
5  5  1.414214       NaN           NaN -4.440892e-16        5  2.220446e-16
```

### `newton(f, df, p0, tol=1e-12, max_iter=100, p_star=None)`

Newton's method, `p_(n+1) = p_n - f(p_n) / f'(p_n)`; `df` is the derivative function. Stops when
`abs(p_(n+1) - p_n) < tol`. Columns: `n, p, f(p), f'(p), step, f_evals, df_evals` (+ `error, q`).
`q` estimates the order of convergence (about 2 for Newton).

```python
from numericalanalysisiiser import newton

t = newton(f, lambda x: 2 * x, 1.0, p_star=math.sqrt(2))
print(t[["n", "p", "f(p)", "step", "error", "q"]])
print("root:", t["p"].iloc[-1], " iterations:", len(t) - 1)
```
```
   n         p          f(p)          step         error         q
0  0  1.000000 -1.000000e+00  5.000000e-01  4.142136e-01       NaN
1  1  1.500000  2.500000e-01  8.333333e-02  8.578644e-02  2.257517
2  2  1.416667  6.944444e-03  2.450980e-03  2.453104e-03  1.983919
3  3  1.414216  6.007305e-06  2.123900e-06  2.123901e-06  1.999767
4  4  1.414214  4.510614e-12  1.594724e-12  1.594724e-12       NaN
5  5  1.414214  4.440892e-16  2.220446e-16  0.000000e+00       NaN
6  6  1.414214 -4.440892e-16           NaN  2.220446e-16       NaN
root: 1.414213562373095  iterations: 6
```

`q` is `NaN` once the error is at round-off level (about 1e-15), where it says nothing.

### `secant(f, p0, p1, tol=1e-12, max_iter=100, p_star=None)`

Secant method: like Newton but without the derivative, using the two latest points. Stops when
`abs(p_(n+1) - p_n) < tol`. Columns: `n, p_prev, p, f(p), p_next, step, f_evals` (+ `error, q`).
`q` is about 1.618 for the secant method (only meaningful once the errors are small).

```python
from numericalanalysisiiser import secant

t = secant(f, 1, 2, tol=1e-10, p_star=math.sqrt(2))
print(t[["n", "p", "p_next", "error", "q"]])
```
```
   n         p    p_next         error         q
0  1  2.000000  1.333333  5.857864e-01 -5.713032
1  2  1.333333  1.400000  8.088023e-02  0.878174
2  3  1.400000  1.414634  1.421356e-02  2.024593
3  4  1.414634  1.414211  4.205840e-04  1.502252
4  5  1.414211  1.414214  2.123898e-06  1.666619
5  6  1.414214  1.414214  3.157747e-10       NaN
6  7  1.414214  1.414214  2.220446e-16       NaN
7  8  1.414214       NaN  0.000000e+00       NaN
```

### Comparing the methods

The last row of each table gives the final result, so a comparison is one small table:

```python
import pandas as pd

runs = {"bisection": bisection(f, 1, 2, tol=1e-10, p_star=math.sqrt(2)),
        "newton": newton(f, lambda x: 2 * x, 1.0, tol=1e-10, p_star=math.sqrt(2)),
        "secant": secant(f, 1, 2, tol=1e-10, p_star=math.sqrt(2))}

last = pd.DataFrame([t.iloc[-1] for t in runs.values()], index=runs.keys())
print(last[["n", "p", "f_evals", "error"]])
```

## Linear systems

`jacobi`, `gauss_seidel` and `sor` solve `A x = b` by iteration. `A` is a square matrix (a list of lists or a numpy
array) with a non-zero diagonal; the methods converge for example if `A` is strictly diagonally dominant.
They stop when `max|x_k - x_(k-1)| < tol` or after `max_iter` iterations.

Columns of the table (one row per iteration; row `k = 0` is the starting vector):

| Column | Meaning |
|---|---|
| `k` | iteration number |
| `omega` | relaxation parameter (SOR only) |
| `step` | `max abs(x_k - x_(k-1))` |
| `residual` | `max abs(b - A x_k)` |
| `q` | `residual_k / residual_(k-1)`; its average over the last rows estimates the convergence factor |
| `x1, x2, ...` | the iterate; **the solution is the last row of these columns** |

The examples solve `4x + y = 1`, `x + 3y = 2`, whose solution is `x = 1/11 = 0.0909...`, `y = 7/11 = 0.6363...`.

### `jacobi(A, b, x0=None, tol=1e-6, max_iter=100)`

Jacobi iteration; `x0` is the starting vector (zeros by default).

```python
from numericalanalysisiiser import jacobi

t = jacobi([[4, 1], [1, 3]], [1, 2], tol=1e-4)
print(t)
solution = t.filter(regex=r"^x\d+$").iloc[-1].to_numpy()
print("solution:", solution, " iterations:", t["k"].iloc[-1])
```
```
   k      step  residual         q        x1        x2
0  0       NaN  2.000000       NaN  0.000000  0.000000
1  1  0.666667  0.666667  0.333333  0.250000  0.666667
2  2  0.166667  0.166667  0.250000  0.083333  0.583333
3  3  0.055556  0.055556  0.333333  0.104167  0.638889
4  4  0.013889  0.013889  0.250000  0.090278  0.631944
5  5  0.004630  0.004630  0.333333  0.092014  0.636574
6  6  0.001157  0.001157  0.250000  0.090856  0.635995
7  7  0.000386  0.000386  0.333333  0.091001  0.636381
8  8  0.000096  0.000096  0.250000  0.090905  0.636333
solution: [0.09090471 0.63633295]  iterations: 8
```

### `gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100)`

Like Jacobi, but each component immediately uses the newest values; it usually needs fewer iterations.

```python
from numericalanalysisiiser import gauss_seidel

print(gauss_seidel([[4, 1], [1, 3]], [1, 2], tol=1e-4))
```
```
   k      step  residual         q        x1        x2
0  0       NaN  2.000000       NaN  0.000000  0.000000
1  1  0.583333  0.583333  0.291667  0.250000  0.583333
2  2  0.145833  0.048611  0.083333  0.104167  0.631944
3  3  0.012153  0.004051  0.083333  0.092014  0.635995
4  4  0.001013  0.000338  0.083333  0.091001  0.636333
5  5  0.000084  0.000028  0.083333  0.090917  0.636361
```

### `sor(A, b, omega, x0=None, tol=1e-6, max_iter=1000)`

Successive over-relaxation with parameter `omega` (`omega = 1` is Gauss-Seidel; convergence needs `0 < omega < 2`).
To compare several values of `omega`, stack the tables:

```python
import pandas as pd
from numericalanalysisiiser import sor

A, b = [[4, 1], [1, 3]], [1, 2]
many = pd.concat([sor(A, b, w, tol=1e-6) for w in [0.8, 1.0, 1.1, 1.2]])
print(many.groupby("omega").last()[["k", "step", "residual"]])
```
```
        k          step      residual
omega                               
0.8    13  7.056235e-07  1.176045e-06
1.0     7  5.860715e-07  1.953572e-07
1.1     7  6.183063e-07  7.409315e-07
1.2    10  7.094322e-07  5.176637e-07
```

`k` is the number of iterations each `omega` needed.

### `tridiagonal_matrix(n, diag=4.0, off=-1.0)`

The `n x n` numpy matrix with `diag` on the main diagonal and `off` on the two neighbouring diagonals.

```python
from numericalanalysisiiser import tridiagonal_matrix

print(tridiagonal_matrix(3))
```
```
[[ 4. -1.  0.]
 [-1.  4. -1.]
 [ 0. -1.  4.]]
```

## Differential equations

### `euler(f, t0, T, y0, h, exact=None)`

Explicit Euler method for the single equation `y' = f(t, y)`, `y(t0) = y0`, on `[t0, T]`. Note that `f` takes two
arguments. `h` is a step size or a list of them; `T - t0` must be a multiple of every `h`. With `exact=` (the exact
solution `y(t)`) you also get `exact` and `error`.

Columns: `h, t, y, stable` (+ `exact, error`). All step sizes are stacked in one table. `stable` is `False`,
and `y` becomes `NaN`, from the moment the solution exceeds `1e10` in absolute value.

```python
import math
from numericalanalysisiiser import euler

# y' = -2y, y(0) = 1, exact solution exp(-2t)
t = euler(lambda t, y: -2 * y, 0, 1, 1.0, [0.5, 0.25], exact=lambda t: math.exp(-2 * t))
print(t[t["h"] == 0.25])
```
```
      h     t       y  stable     exact     error
3  0.25  0.00  1.0000    True  1.000000  0.000000
4  0.25  0.25  0.5000    True  0.606531  0.106531
5  0.25  0.50  0.2500    True  0.367879  0.117879
6  0.25  0.75  0.1250    True  0.223130  0.098130
7  0.25  1.00  0.0625    True  0.135335  0.072835
```

One line per step size, with the largest error and the stability:

```python
print(t.groupby("h", sort=False).agg(rows=("t", "count"), max_error=("error", "max"), stable=("stable", "all")))
```
```
      rows  max_error  stable
h                            
0.50     3   0.367879    True
0.25     5   0.117879    True
```

(`rows` is the number of steps plus one.) The error roughly halves when `h` is halved: Euler is a first-order method.

For a stiff problem the explicit method becomes unstable when `h` is too large:

```python
fs = lambda t, y: -1000 * (y - math.cos(t)) - math.sin(t)      # exact solution cos(t)
s = euler(fs, 0, 1, 1.0, [0.01, 0.002], exact=math.cos)
print(s.groupby("h", sort=False).agg(rows=("t", "count"), max_error=("error", "max"), stable=("stable", "all")))
```
```
       rows     max_error  stable
h                                
0.010   101  9.265092e+09   False
0.002   501  1.999999e-06    True
```

---

# Plotting

The library does not plot. Take columns from the table; for example the error of the bisection example:

```python
import matplotlib.pyplot as plt

t = bisection(f, 1, 2, tol=1e-10, p_star=math.sqrt(2))
plt.semilogy(t["n"], t["error"])
plt.xlabel("iteration n")
plt.ylabel("error")
plt.show()
```

(`pip install matplotlib` if you do not have it.) For errors against the step size use `plt.loglog(t["h"], t["central_error"])`.

# Notes

* Interpolating with many equally spaced nodes (`lagrange`, `newton_interpolation`, `hermite`) is numerically unstable (Runge phenomenon).
* `euler` handles a single scalar equation.
* The linear solvers use plain Python loops, so they are meant for small and medium systems.
* `newton` needs `f'(p)` different from zero at every iterate; `secant` stops early if two successive function values are equal.
* To get the exact value for `exact=` / `p_star=` you can use `scipy.optimize.brentq` (roots) or `scipy.integrate.quad` (integrals).

# License

MIT, see [LICENSE](LICENSE).
