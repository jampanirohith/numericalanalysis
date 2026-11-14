# Numerical Analysis

Course work and a Python library for **MAT 308: Mathematics Lab** and **MAT 314: Numerical Analysis**
(School of Mathematics, IISER Thiruvananthapuram).

The lab scripts are in [`College/`](College). The methods from the labs were collected into a Python
library, [`numericalanalysisiiser`](https://pypi.org/project/numericalanalysisiiser/), which is in
[`pythonlibrary/`](pythonlibrary).

```bash
pip install numericalanalysisiiser
```

```python
from numericalanalysisiiser import bisection

t = bisection(lambda x: x**3 - x - 2, 1, 2, tol=1e-10)
print(t["p"].iloc[-1])        # the root: last row of the table
```

**Full documentation (every function, its parameters, the columns it returns, and examples with output):
[pythonlibrary/README.md](pythonlibrary/README.md).**

## Repository layout

| Folder | Contents |
|---|---|
| [`College/Lab`](College/Lab) | the lab scripts, one folder per lab session |
| [`College/Practice Problems`](College/Practice%20Problems) | the lab problem sheets (Lab 1 to Lab 7) with the solution scripts |
| [`College/Notes`](College/Notes) | lecture / lab notes |
| [`pythonlibrary`](pythonlibrary) | the installable Python package, its README, license and `pyproject.toml` |

## Course syllabus

### MAT 314: Numerical Analysis [3 0 0 3]

This introductory numerics course makes students aware of the classical approximation schemes used to
solve algebraic equations and differential equations. Prerequisites: none.

| Topic | In the library |
|---|---|
| Round-off errors and computer arithmetic | not implemented (done in the lab scripts) |
| Nonlinear equations: bisection, secant and Newton's method, fixed-point iteration | `bisection`, `secant`, `newton`, `fixed_point` |
| Interpolation: Lagrange, divided differences, Hermite, splines | `lagrange`, `newton_interpolation`, `divided_differences`, `hermite`, `natural_cubic_spline` |
| Numerical differentiation, Richardson extrapolation | `finite_difference`, `richardson`, `richardson_triangle` |
| Numerical integration: trapezoidal, Simpson, Newton-Cotes, Gauss quadrature, Romberg | `trapezoid`, `simpson`, `gauss_legendre`, `romberg` (trapezoid and Simpson are the Newton-Cotes rules of degree 1 and 2) |
| Linear systems, direct methods: Gauss elimination, pivoting, matrix factorisations | not implemented |
| Linear systems, iterative methods: matrix norms, Jacobi and Gauss-Seidel, relaxation | `jacobi`, `gauss_seidel`, `sor` (matrix norms not implemented) |
| Eigenvalues and eigenvectors: power method, QR algorithm | not implemented |
| Initial value problems: Euler method, higher-order Runge-Kutta methods | `euler` (Runge-Kutta not implemented) |
| Boundary value problems: finite differences | not implemented |

Reference books: K. E. Atkinson, *An Introduction to Numerical Analysis*, 2nd ed.; R. L. Burden and
J. D. Faires, *Numerical Analysis*, 7th ed.

### MAT 308: Mathematics Lab [0 0 3 1]

The aim is to familiarise students with the implementation of numerical analysis techniques, especially
numerical integration and solving linear systems of equations, through a set of weekly lab assignments
in Python / MATLAB. Prerequisites: none.

| Lab | Topic | In the library |
|---|---|---|
| 1 | Introduction to Python / MATLAB | |
| 2 | Computer arithmetic: machine epsilon; error of a forward difference against the step size `h` | `finite_difference` covers the second part |
| 3 | Root finding: Newton's method and fixed-point iteration | `newton`, `fixed_point` (also `bisection`, `secant`) |
| 4 | Interpolation: Lagrange polynomial of degree 10 for `1 / (1 + x^2)` on `[-5, 5]` | `lagrange` (also `newton_interpolation`, `hermite`, `natural_cubic_spline`) |
| 5 | Numerical integration with the composite Simpson and trapezoidal rules (inside Newton's method) | `simpson`, `trapezoid` (also `romberg`, `gauss_legendre`) |
| 6 | LU decomposition of given matrices | not implemented |
| 7 | Gauss-Seidel and Jacobi methods for `Ax = b` | `gauss_seidel`, `jacobi` (also `sor`) |
| 8 | Initial value problems: Runge-Kutta of order 2 and 4 | not implemented (`euler` only) |
| 9 | Boundary value problems: nonlinear finite-difference method | not implemented |
| 10 | Power method for the dominant eigenvalue | not implemented |

Reference books: D. Mitsotakis, *An Introduction to Numerical Analysis and Scientific Computing with Python*,
CRC Press, 2023; R. L. Burden and J. D. Faires, *Numerical Analysis*, 7th ed.; S. D. Conte and C. de Boor,
*Elementary Numerical Analysis: An Algorithmic Approach*, 3rd ed.

## The Python library

`numericalanalysisiiser` collects the numerical methods from the labs. **Every function returns a pandas
DataFrame**, with one row per iteration, step size or evaluation point, so you can pick any columns and plot
them yourself. The optional `exact=` / `p_star=` argument adds `error` columns and the observed order of
convergence.

### Functions

| Area | Functions | What they do |
|---|---|---|
| Interpolation | `lagrange(x, y, alpha)` | Lagrange polynomial through the points |
| | `newton_interpolation(x, y, alpha)` | Newton form of the interpolation polynomial |
| | `divided_differences(x, y)` | divided difference table (first row = Newton coefficients) |
| | `hermite(x, y, dy, alpha)` | interpolation matching values and slopes |
| | `natural_cubic_spline(x, y, alpha)` | natural cubic spline |
| Differentiation | `finite_difference(f, x, h)` | forward, backward, central and second-derivative formulas with error and order |
| | `richardson(f, x, h)` | Richardson extrapolation, by step size or as the triangle table |
| | `richardson_triangle(first_column)` | the Richardson triangle for your own values |
| Integration | `trapezoid(f, a, b, n)` | composite trapezoidal rule |
| | `simpson(f, a, b, n)` | composite Simpson 1/3 rule |
| | `romberg(f, a, b)` | Romberg table, by number of levels or tolerance |
| | `gauss_legendre(f, a, b, n)` | Gauss-Legendre quadrature with `n` points |
| | `composite_trapezoid`, `composite_simpson` | the same rules as a single number |
| Root finding | `bisection(f, a, b)` | bisection with the theoretical error bound |
| | `fixed_point(g, p0)` | fixed-point iteration |
| | `newton(f, df, p0)` | Newton's method with the estimated order `q` |
| | `secant(f, p0, p1)` | secant method with the estimated order `q` |
| Linear systems | `jacobi(A, b)` | Jacobi iteration, with residual history |
| | `gauss_seidel(A, b)` | Gauss-Seidel iteration |
| | `sor(A, b, omega)` | successive over-relaxation |
| | `tridiagonal_matrix(n)` | the 4 / -1 tridiagonal test matrix |
| ODE | `euler(f, t0, T, y0, h)` | explicit Euler method, with a stability flag |

See [pythonlibrary/README.md](pythonlibrary/README.md) for the arguments and returned columns of each
function, worked examples with output, and recipes (comparing root finders, comparing relaxation
parameters, stability tables for Euler).

### Development

```bash
cd pythonlibrary
pip install -e .
```

## License

The library is released under the MIT license, see [pythonlibrary/LICENSE](pythonlibrary/LICENSE).
