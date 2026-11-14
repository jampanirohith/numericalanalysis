# numericalanalysisiiser

Classical numerical analysis methods in plain Python, from the IISER Thiruvananthapuram numerical analysis labs (MAT 308 / MAT 314): interpolation, numerical differentiation and
integration, root finding, iterative linear solvers and an ODE solver.

**Every method returns a pandas DataFrame.** The table has one row per iteration, step size or
evaluation point, so you can read the result, filter it, and plot any columns you like.

```python
from numericalanalysisiiser import bisection
import matplotlib.pyplot as plt

f = lambda x: x**3 - x - 2
t = bisection(f, 1, 2, tol=1e-10, p_star=1.5213797068045676)

print(t["p"].iloc[-1])                    # the root (last row)
plt.semilogy(t["n"], t["error"])          # any columns can be plotted
```

## Contents

* [Installation](#installation)
* [How the functions behave](#how-the-functions-behave)
* [Interpolation](#interpolation) - `lagrange`, `newton_interpolation`, `divided_differences`, `hermite`, `natural_cubic_spline`
* [Numerical differentiation](#numerical-differentiation) - `finite_difference`, `richardson`, `richardson_triangle`
* [Numerical integration](#numerical-integration) - `trapezoid`, `simpson`, `romberg`, `gauss_legendre`
* [Root finding](#root-finding) - `bisection`, `fixed_point`, `newton`, `secant`
* [Linear systems](#linear-systems) - `jacobi`, `gauss_seidel`, `sor`, `tridiagonal_matrix`
* [Ordinary differential equations](#ordinary-differential-equations) - `euler`
* [Recipes](#recipes)
* [Notes and limitations](#notes-and-limitations)
* [License](#license)

## Installation

```bash
pip install numericalanalysisiiser
```

Package on PyPI: <https://pypi.org/project/numericalanalysisiiser/>. Source code and the course material
it comes from: <https://github.com/jampanirohith/numericalanalysis>.

Requires Python 3.9 or newer, numpy and pandas (installed automatically). matplotlib is not
required; install it only if you want to plot the tables.

To upgrade to the latest release:

```bash
pip install --upgrade numericalanalysisiiser
```

To work on the code from a clone of the repository:

```bash
cd pythonlibrary
pip install -e .
```

## How the functions behave

**The functions.** `f`, `g`, `df` and `exact` are ordinary Python functions of one number
(`f(x)`; ODE right-hand sides take two: `f(t, y)`). Functions from `math` and from `numpy` both work.

**Lists in, rows out.** Wherever a quantity can vary (`alpha`, `h`, `n`, `omega`), you may pass one
value or a list. You get one row per value.

**Optional exact values.** `exact=` (or `p_star=` for root finders) is the true value. When given,
the table gets `error` columns, and `order` (observed order of convergence) or `q` (estimated order
of an iteration) where they make sense.

**Iterative methods.** Root finders and linear solvers return one row per iteration. **The last
row is the final answer.**

**`order` columns.** `order` in row `k` is `log(E_k / E_(k+1)) / log(h_k / h_(k+1))`, computed from a
row and the next one, so the last row is `NaN`. It needs the step sizes to be listed in
decreasing order.

**`NaN` values** mean "not defined for this row", never an error.

## Interpolation

All five functions return the columns `alpha, p` (and `exact, error` when `exact` is given), where
`p` is the interpolant evaluated at `alpha`.

### `lagrange(x, y, alpha, exact=None)`

The Lagrange interpolating polynomial through the points `(x_i, y_i)`:
`p(a) = sum_i y_i * prod_(j != i) (a - x_j) / (x_i - x_j)`.

| Parameter | |
|---|---|
| `x`, `y` | nodes and values (lists or arrays); the `x` values must be distinct |
| `alpha` | a number or a list of points where the polynomial is evaluated |
| `exact` | optional function `f(x)`; adds `exact` and `error = abs(exact - p)` |

```python
from numericalanalysisiiser import lagrange

lagrange([-2, 0, -1, 1, 2], [4, 1, -1, 1, 9], [0.7, 1.5], exact=lambda t: t**2)
```
```
 alpha        p  exact    error
   0.7 0.982413   0.49 0.492413
   1.5 2.632812   2.25 0.382812
```

### `newton_interpolation(x, y, alpha, exact=None)`

The same polynomial in Newton form, `c0 + c1 (a - x0) + c2 (a - x0)(a - x1) + ...`, with divided
difference coefficients. Same parameters and columns as `lagrange`. The nodes may be in any order.

### `divided_differences(x, y)`

The divided difference table. Columns `x, f[0], f[1], ...`; the **first row holds the Newton
coefficients** `c0, c1, ...`.

```python
from numericalanalysisiiser import divided_differences

divided_differences([0, 1, 2, 3], [1, 2, 5, 10])
```
```
 x  f[0]  f[1]  f[2]  f[3]
 0     1   1.0   1.0   0.0
 1     2   3.0   1.0   NaN
 2     5   5.0   NaN   NaN
 3    10   NaN   NaN   NaN
```

### `hermite(x, y, dy, alpha, exact=None)`

Hermite interpolation: the polynomial of degree `2n - 1` that matches both the values `y_i` and the
slopes `dy_i` at the `n` nodes. Same columns as `lagrange`.

```python
import numpy as np
from numericalanalysisiiser import hermite

f  = lambda x: 1 / (1 + 25 * x**2)
df = lambda x: -50 * x / (1 + 25 * x**2) ** 2
x = np.linspace(-1, 1, 10)
t = hermite(x, f(x), df(x), np.linspace(-1, 1, 2001), exact=f)
t["error"].max()
```

### `natural_cubic_spline(x, y, alpha, exact=None)`

The natural cubic spline (second derivative zero at both ends). `x` must be in increasing order.
Outside `[x[0], x[-1]]` the first / last cubic piece is extended. Same columns as `lagrange`.

```python
from numericalanalysisiiser import natural_cubic_spline

f = lambda t: np.exp(-t) * np.sin(2 * t)
for h in [0.5, 0.25, 0.125]:
    nodes = np.arange(0, 4 + h / 2, h)
    err = natural_cubic_spline(nodes, f(nodes), np.linspace(0, 4, 2001), exact=f)["error"].max()
    print(h, err)
```

## Numerical differentiation

### `finite_difference(f, x, h, exact=None, methods=("forward", "backward", "central"))`

Finite difference approximations at the point `x`.

| Method | Formula | Error |
|---|---|---|
| `"forward"` | `(f(x+h) - f(x)) / h` | O(h) |
| `"backward"` | `(f(x) - f(x-h)) / h` | O(h) |
| `"central"` | `(f(x+h) - f(x-h)) / (2h)` | O(h^2) |
| `"second"` | `(f(x+h) - 2 f(x) + f(x-h)) / h^2`, approximates `f''(x)` | O(h^2) |

| Parameter | |
|---|---|
| `h` | a step size or a list of step sizes |
| `exact` | the true derivative at `x`; adds `<method>_error` and `<method>_order` |
| `methods` | tuple of method names |

Returns `h`, then one column per method (and its `_error`, `_order`).

```python
import math
from numericalanalysisiiser import finite_difference

f = lambda x: math.exp(x) * math.cos(x)
exact = math.exp(1) * (math.cos(1) - math.sin(1))
finite_difference(f, 1.0, [0.1, 0.05, 0.025], exact)
```
```
    h   forward  forward_error  forward_order  backward  backward_error  backward_order   central  central_error  central_order
0.100 -1.060159       0.241497       1.039001 -0.602199        0.216463        0.960073 -0.831179       0.012517       1.999763
0.050 -0.936190       0.117528       1.019624 -0.707393        0.111268        0.980145 -0.821791       0.003130       1.999941
0.025 -0.876632       0.057970            NaN -0.762256        0.056405             NaN -0.819444       0.000782            NaN
```

### `richardson(f, x, h, levels=1, method="central", exact=None, layout="by_h")`

Richardson extrapolation of a finite difference formula `D(h)`, which removes the leading error terms.
For `"central"` and `"second"` the error series is `h^2, h^4, ...` (factor `4^j`); for `"forward"` and
`"backward"` it is `h, h^2, ...` (factor `2^j`).

`layout="by_h"` (default): `h` is a list of step sizes, one row per `h`. Each row uses `D` at
`h, h/2, ..., h/2^levels`:

```
R0 = D(h)
R1 = (4 D(h/2) - D(h)) / 3                 (central)
R2 = (16 R1(h/2) - R1(h)) / 15
```

Returns `h, R0, R1, ..., R<levels>`.

```python
from numericalanalysisiiser import richardson

richardson(f, 1.0, [0.4, 0.2, 0.1], levels=1, exact=exact)
```
```
  h        R0        R1  R0_error  R0_order     R1_error  R1_order
0.4 -1.018261 -0.818844  0.199600  1.996053 1.822700e-04  4.045968
0.2 -0.868698 -0.818672  0.050037  1.999046 1.103000e-05  4.011736
0.1 -0.831179 -0.818662  0.012517       NaN 6.800000e-07       NaN
```

`layout="triangle"`: `h` is a **single** starting step `h0`; row `k` uses `h0 / 2^k` and the table is
the classical triangle `R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1]) / (4^j - 1)`.
`levels + 1` rows. Returns `k, h, R0, R1, ...`; the entries above the diagonal are `NaN`. With
`exact`, `R<j>_order` is the observed order of column `j` from row to row.

```python
richardson(f, 1.0, 1.0, levels=3, layout="triangle")
```
```
 k     h        R0        R1        R2        R3
 0 1.000 -2.037466       NaN       NaN       NaN
 1 0.500 -1.129867 -0.827334       NaN       NaN
 2 0.250 -0.896804 -0.819117 -0.818569       NaN
 3 0.125 -0.838217 -0.818688 -0.818660 -0.818661
```

### `richardson_triangle(first_column, p=2)`

The triangle on its own, for values you computed yourself. `first_column[k]` must be computed with
step `h / 2^k`. Returns a list of rows (`R[k][j]`). Use `p=2` for central differences and the
trapezoidal rule, `p=1` for forward / backward differences.

## Numerical integration

### `trapezoid(f, a, b, n=1, exact=None)` and `simpson(f, a, b, n=2, exact=None)`

Composite trapezoidal rule and composite Simpson 1/3 rule for the integral of `f` over `[a, b]`.

| Parameter | |
|---|---|
| `n` | number of sub-intervals, or a list of them. `n=1` (trapezoid) and `n=2` (Simpson) are the basic single-application rules. For Simpson `n` must be even, otherwise `ValueError` |
| `exact` | the true integral; adds `error` and `order` |

Returns `n, h, value` (and `error, order`), where `h = (b - a) / n`.

```python
import math
from numericalanalysisiiser import trapezoid, simpson

g = lambda x: math.exp(x)
trapezoid(g, 0, 1, [2, 4, 8], exact=math.e - 1)
```
```
 n     h    value    error    order
 2 0.500 1.753931 0.035649 1.995513
 4 0.250 1.727222 0.008940 1.998874
 8 0.125 1.720519 0.002237      NaN
```
```python
simpson(g, 0, 1, [2, 4, 8], exact=math.e - 1)
```
```
 n     h    value    error    order
 2 0.500 1.718861 0.000579 3.968247
 4 0.250 1.718319 0.000037 3.991978
 8 0.125 1.718284 0.000002      NaN
```

The observed orders are about 2 for the trapezoidal rule and about 4 for Simpson's rule.

`composite_trapezoid(f, a, b, n)` and `composite_simpson(f, a, b, n)` return the single number.

### `romberg(f, a, b, levels=6, tol=None, exact=None)`

Romberg integration. Column `R0` is the trapezoidal rule with `n = 1, 2, 4, 8, ...` sub-intervals; the other
columns are Richardson extrapolations of it (`R[k][j] = R[k][j-1] + (R[k][j-1] - R[k-1][j-1]) / (4^j - 1)`).

| Parameter | |
|---|---|
| `levels` | number of rows (row `k` uses `n = 2^k`) |
| `tol` | if given, rows are added until `abs(R[k][k] - R[k-1][k-1]) < tol` (at most 25 rows); `levels` is then ignored |
| `exact` | the true integral; adds `error`, the error of the diagonal entry `R[k][k]` of each row |

Returns `n, R0, R1, ...` (and `error`). **The answer is the last entry of the last row.**

```python
from numericalanalysisiiser import romberg

romberg(g, 0, 1, levels=4, exact=math.e - 1)
```
```
 n       R0       R1       R2       R3        error
 1 1.859141      NaN      NaN      NaN 1.408591e-01
 2 1.753931 1.718861      NaN      NaN 5.793234e-04
 4 1.727222 1.718319 1.718283      NaN 8.595000e-07
 8 1.720519 1.718284 1.718282 1.718282 3.000000e-10
```

### `gauss_legendre(f, a, b, n=2, exact=None)`

Gauss-Legendre quadrature with `n` points (nodes and weights from numpy, mapped from `[-1, 1]` to
`[a, b]`). An `n`-point rule is exact for polynomials up to degree `2n - 1`.

`n` may be a list. Returns `n, value` (and `error`).

```python
from numericalanalysisiiser import gauss_legendre

gauss_legendre(g, 0, 1, [1, 2, 3], exact=math.e - 1)
```
```
 n    value        error
 1 1.648721 6.956056e-02
 2 1.717896 3.854505e-04
 3 1.718281 8.240870e-07
```

## Root finding

All four methods share these conventions:

* One row per iteration; the last row is the final iterate, so **the root is `t["p"].iloc[-1]`**.
* `p_star` is the exact root. Only when you pass it do the tables get `error` (and `q`).
* `f_evals` / `df_evals` are the cumulative numbers of evaluations of `f` / `f'` the method made.
* `q = log(e_(n+1) / e_n) / log(e_n / e_(n-1))` is the estimated order of convergence (about 2 for Newton, 1.618 for secant). It is only meaningful once the errors are small, and is `NaN` where the error is at round-off level.
* Number of iterations: `len(t)` for `bisection`; `len(t) - 1` for `fixed_point`, `newton` and `secant` (their last row only shows the final iterate).

### `bisection(f, a, b, tol=1e-10, max_iter=1000, p_star=None)`

Bisection on `[a, b]`. `f(a)` and `f(b)` must have opposite signs, otherwise `ValueError`. Stops when
`(b - a) / 2 < tol`, which guarantees an error below `tol`.

Columns: `n, a, b, p, f(p), b-a, bound, f_evals` (+ `error`). `bound = (b0 - a0) / 2^(n+1)` is the
theoretical error bound after iteration `n`. The number of iterations needed for the error to be below `tol` is
`math.ceil(math.log2((b - a) / tol))`.

```python
from numericalanalysisiiser import bisection

f = lambda x: x**3 - x - 2
t = bisection(f, 1, 2, tol=1e-3, p_star=1.5213797068045676)
t.head(4)
```
```
 n   a     b      p      f(p)   b-a  bound  f_evals   error
 0 1.0 2.000 1.5000 -0.125000 1.000 0.5000        3 0.02138
 1 1.5 2.000 1.7500  1.609375 0.500 0.2500        4 0.22862
 2 1.5 1.750 1.6250  0.666016 0.250 0.1250        5 0.10362
 3 1.5 1.625 1.5625  0.252197 0.125 0.0625        6 0.04112
```

### `fixed_point(g, p0, tol=1e-10, max_iter=1000, f=None, p_star=None)`

Fixed point iteration `p_(n+1) = g(p_n)`; stops when `abs(p_(n+1) - p_n) < tol`.
`g` converges near a fixed point `p*` when `abs(g'(p*)) < 1`.

| Parameter | |
|---|---|
| `f` | optional original function; adds the residual column `f(p)` |
| `p_star` | the exact **fixed point** of `g` (not necessarily a root of `f`); adds `error` |

Columns: `n, p, g(p), step, [f(p)], f_evals` (+ `error`). If `g` raises an error, returns a complex
number, or the values exceed `1e12`, the iteration stops there. If it never converges you get
`max_iter + 1` rows.

```python
from numericalanalysisiiser import fixed_point

F = lambda x: x**3 - 2 * x - 5
t = fixed_point(lambda x: (2 * x + 5) ** (1 / 3), 2.5, tol=1e-6, f=F, p_star=2.0945514815423265)
```
```
 n        p     g(p)         step     f(p)  f_evals        error
 0 2.500000 2.154435 3.455653e-01 5.625000        1 4.054485e-01
 1 2.154435 2.103612 5.082266e-02 0.691131        2 5.988321e-02
 ...
 7 2.094552 2.094552 6.200000e-07 0.000008        8 7.300000e-07
 8 2.094552      NaN          NaN 0.000001        8 1.100000e-07
```

### `newton(f, df, p0, tol=1e-12, max_iter=100, p_star=None)`

Newton's method `p_(n+1) = p_n - f(p_n) / f'(p_n)`; stops when `abs(p_(n+1) - p_n) < tol`.
`df` is the derivative function and must not be zero at an iterate.

Columns: `n, p, f(p), f'(p), step, f_evals, df_evals` (+ `error, q`).

```python
from numericalanalysisiiser import newton

t = newton(lambda x: math.exp(-x) - x, lambda x: -math.exp(-x) - 1, 0.5, p_star=0.5671432904097838)
```
```
 n        p          f(p)     f'(p)         step  f_evals  df_evals        error        q
 0 0.500000  1.065307e-01 -1.606531 6.631100e-02        1         1 6.714329e-02      NaN
 1 0.566311  1.304510e-03 -1.567616 8.321600e-04        2         2 8.322900e-04 2.004512
 2 0.567143  2.000000e-07 -1.567143 1.300000e-07        3         3 1.300000e-07 2.002809
 3 0.567143  0.000000e+00 -1.567143 0.000000e+00        4         4 0.000000e+00      NaN
 4 0.567143 -0.000000e+00       NaN          NaN        4         4 0.000000e+00      NaN
```

### `secant(f, p0, p1, tol=1e-12, max_iter=100, p_star=None)`

Secant method (no derivative): `p_(n+1) = p_n - f(p_n) (p_n - p_(n-1)) / (f(p_n) - f(p_(n-1)))`.
Stops when `abs(p_(n+1) - p_n) < tol`, or early if `f(p_n) == f(p_(n-1))`.

Columns: `n, p_prev, p, f(p), p_next, step, f_evals` (+ `error, q`). Here `error` is `abs(p - p_star)`.

```python
from numericalanalysisiiser import secant

g = lambda x: x**3 - 2 * x - 5
secant(g, 2, 3, tol=1e-10, p_star=2.0945514815423265)
```
```
 n   p_prev        p      f(p)   p_next     step  f_evals    error         q
 1 2.000000 3.000000 16.000000 2.058824 0.941176        3 0.905449 -1.430760
 2 3.000000 2.058824 -0.390800 2.081264 0.022440        4 0.035728  0.305982
 3 2.058824 2.081264 -0.147204 2.094824 0.013560        5 0.013288  3.929247
 4 2.081264 2.094824  0.003044 2.094549 0.000275        6 0.000273  1.258286
 5 2.094824 2.094549 -0.000023 2.094551 0.000002        7 0.000002  1.795831
 ...
```

## Linear systems

Solve `A x = b` with stationary iterations. `A` is a square matrix (nested lists or a numpy array) and
`b` a vector. `A` needs a non-zero diagonal; the methods are guaranteed to converge for example when
`A` is strictly diagonally dominant. The solvers loop in pure Python, which is fine for small and
medium systems.

All three return one row per iteration and stop when `max|x_k - x_(k-1)| < tol` or after `max_iter` iterations.
Columns:

| Column | |
|---|---|
| `k` | iteration number; row `0` is the starting vector |
| `omega` | the relaxation parameter (SOR only) |
| `step` | `max|x_k - x_(k-1)|` (`NaN` in row 0) |
| `residual` | `max|b - A x_k|` |
| `q` | `residual_k / residual_(k-1)`; its average over the last rows estimates the asymptotic convergence factor |
| `x1, x2, ...` | the iterate `x_k`; **the solution is the last row** |

### `jacobi(A, b, x0=None, tol=1e-6, max_iter=100)`

`x_i^(k+1) = (b_i - sum_(j != i) a_ij x_j^(k)) / a_ii`. `x0` is the starting vector (zeros by default).

```python
from numericalanalysisiiser import jacobi

A = [[10, -1, 2, 1], [-1, 11, -1, 3], [2, -1, 10, -1], [1, 3, -1, 8]]
b = [6, 25, -11, 15]
t = jacobi(A, b, tol=1e-4)
t.head(4)
```
```
 k     step  residual        q       x1       x2        x3       x4
 0      NaN 25.000000      NaN 0.000000 0.000000  0.000000 0.000000
 1 2.272727  8.518182 0.340727 0.600000 2.272727 -1.100000 1.875000
 2 1.064773  3.748864 0.440101 0.859773 1.715909 -0.805227 0.810227
 3 0.340806  1.228372 0.327665 0.851614 2.056715 -1.019341 1.023409
```

The solution vector and the number of iterations:

```python
solution = t.filter(regex=r"^x\d+$").iloc[-1].to_numpy()
iterations = t["k"].iloc[-1]
```

### `gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100)`

Like Jacobi, but each component uses the already updated components: `x_i^(k+1) = (b_i - sum_(j<i) a_ij x_j^(k+1) - sum_(j>i) a_ij x_j^(k)) / a_ii`. Same columns as `jacobi`.

### `sor(A, b, omega, x0=None, tol=1e-6, max_iter=1000)`

Successive over-relaxation: `x_i^(k+1) = (1 - omega) x_i^(k) + omega * (Gauss-Seidel value)`.
`omega = 1` is Gauss-Seidel; `0 < omega < 2` is required for convergence.
`omega` is one number per call; to compare several values, stack the tables:

```python
import pandas as pd
from numericalanalysisiiser import sor

A = [[4, 1, -1, 1], [1, 4, -1, -1], [-1, -1, 5, 1], [1, -1, 1, 3]]
b = [-2, -1, 0, 1]
t = pd.concat([sor(A, b, w, tol=1e-3) for w in [0.9, 1.0, 1.1, 1.5]])
t.groupby("omega").last()[["k", "step", "residual"]]
```
```
        k      step  residual
omega
0.9     9  0.000971  0.002034
1.0     8  0.000451  0.000641
1.1     6  0.000105  0.000157
1.5    11  0.000825  0.001213
```

### `tridiagonal_matrix(n, diag=4.0, off=-1.0)`

The `n x n` numpy matrix with `diag` on the main diagonal and `off` on the first sub- and
super-diagonal.

```python
from numericalanalysisiiser import tridiagonal_matrix

tridiagonal_matrix(4)
```
```
[[ 4. -1.  0.  0.]
 [-1.  4. -1.  0.]
 [ 0. -1.  4. -1.]
 [ 0.  0. -1.  4.]]
```

## Ordinary differential equations

### `euler(f, t0, T, y0, h, exact=None)`

The explicit Euler method `y_(i+1) = y_i + h f(t_i, y_i)` for the scalar problem `y' = f(t, y)`,
`y(t0) = y0` on `[t0, T]`.

| Parameter | |
|---|---|
| `f` | `f(t, y)`, takes two numbers |
| `h` | step size or a list of step sizes; `T - t0` must be a multiple of every `h`, otherwise `ValueError` |
| `exact` | the exact solution `y(t)`; adds `exact` and `error` |

Returns one table with the runs for all step sizes stacked: `h, t, y, stable` (+ `exact, error`).
`stable` is `False` for a run whose solution blew up (`abs(y) > 1e10`); `y` is then `NaN` from that step on.

```python
import math
from numericalanalysisiiser import euler

f = lambda t, y: -1000 * (y - math.cos(t)) - math.sin(t)       # stiff problem, exact solution cos(t)
t = euler(f, 0, 1, 1.0, [0.01, 0.002, 0.001], exact=math.cos)

t.groupby("h", sort=False).agg(N=("t", "count"), max_error=("error", "max"), stable=("stable", "all"))
```
```
          N     max_error  stable
h
0.010   101  9.265092e+09   False
0.002   501  1.999999e-06    True
0.001  1001  5.000000e-07    True
```

(`N` here counts the rows, i.e. the number of steps plus one.) Plot the runs with
`for h, part in t.groupby("h"): plt.plot(part["t"], part["y"], label=h)`.

## Recipes

**Compare root finders (final row of each table):**

```python
import pandas as pd
runs = {"Bisection": bisection(f, 2, 3, p_star=root),
        "Fixed point": fixed_point(g, 2.5, p_star=root),
        "Newton": newton(f, df, 2.5, p_star=root),
        "Secant": secant(f, 2, 3, p_star=root)}
last = pd.DataFrame([t.iloc[-1] for t in runs.values()], index=runs.keys())
last[["n", "p", "f_evals", "df_evals", "error"]]
```

**Error against the number of points or the mesh size.** Evaluate on a fine grid with `exact=` and take
the maximum error:

```python
errors = [natural_cubic_spline(nodes, f(nodes), fine_grid, exact=f)["error"].max() for nodes in node_sets]
```

**Log-log convergence plot of a differencing table:**

```python
t = finite_difference(f, 1.0, [2.0**-k for k in range(1, 11)], exact)
plt.loglog(t["h"], t["forward_error"], label="forward")
plt.loglog(t["h"], t["central_error"], label="central")
```

**Exact values.** For root finders get `p_star` from `scipy.optimize.brentq` or `mpmath.findroot`; for
integrals use `scipy.integrate.quad`.

## Notes and limitations

* Hermite and Lagrange interpolation at many equally spaced nodes is numerically unstable (Runge phenomenon). Use fewer nodes or a spline.
* `natural_cubic_spline` needs increasing `x`.
* `euler` handles one scalar equation, not systems.
* The linear solvers use plain Python loops (`O(n^2)` per iteration) and a dense matrix.
* `newton` raises `ZeroDivisionError` if `f'` is zero at an iterate; `secant` stops early if two consecutive function values are equal.
* Developed and checked with Python 3.12, numpy 2 and pandas 3.

## License

MIT, see [LICENSE](LICENSE).
