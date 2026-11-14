# numericalanalysisiiser

A Python library with the numerical methods from the MAT 308 Mathematics Lab (School of Mathematics,
IISER Thiruvananthapuram). Every method is a short function in its own file, and every function returns a
pandas DataFrame with one row per iteration, step size or evaluation point.

* PyPI: <https://pypi.org/project/numericalanalysisiiser/>
* Source code and documentation: <https://github.com/jampanirohith/numericalanalysis>

## Install

You need Python 3.9 or newer. In a terminal:

```
pip install numericalanalysisiiser
```

Installing the library also installs numpy and pandas. `scipy` is only needed by the check script below.

## Functions

| Area | Functions |
|---|---|
| Interpolation | `lagrange`, `newton_interpolation`, `divided_differences`, `hermite`, `natural_cubic_spline` |
| Numerical differentiation | `finite_difference` (forward, backward, central and second-derivative formulas), `richardson` (Richardson extrapolation) |
| Numerical integration | `trapezoid`, `simpson`, `romberg`, `gauss_legendre` |
| Root finding | `bisection`, `fixed_point`, `newton`, `secant` |
| Iterative linear solvers | `jacobi`, `gauss_seidel`, `sor` |
| Differential equations | `euler` (explicit Euler method, with a stability flag) |

The optional arguments `exact=` (or `p_star=` for the root finders) take the true value and add `error` columns
and the observed order of convergence to the table. For the iterative methods the **last row of the table is the
answer**.

## Two examples

Lagrange interpolation at `x = 0.7`:

```python
from numericalanalysisiiser import lagrange

lagrange([-2, 0, -1, 1, 2], [4, 1, -1, 1, 9], 0.7)
```
```
 alpha        p
   0.7 0.982413
```

Bisection for `x^3 - x - 2 = 0` on `[1, 2]`:

```python
from numericalanalysisiiser import bisection

t = bisection(lambda x: x**3 - x - 2, 1, 2, tol=1e-10)
print(len(t), "iterations; root =", t["p"].iloc[-1])
```
```
34 iterations; root = 1.5213797068572603
```

The columns of `t` are `n, a, b, p, f(p), b-a, bound, f_evals`; adding `p_star=` gives an `error` column as well.