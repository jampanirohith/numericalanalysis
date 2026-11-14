# Numerical Analysis

Lab work and a Python library for **MAT 308: Mathematics Lab** (School of Mathematics, IISER Thiruvananthapuram).

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

To check the lab solutions independently, see [CROSSCHECK.md](CROSSCHECK.md): install the library and run
[`crosscheck.py`](crosscheck.py), which compares the results for Labs 1 to 7 with numpy / scipy references.

## Repository layout

| Folder | Contents |
|---|---|
| [`College/Lab`](College/Lab) | the lab scripts, one folder per lab session |
| [`College/Practice Problems`](College/Practice%20Problems) | the lab problem sheets (Lab 1 to Lab 7) with the solution scripts |
| [`College/Notes`](College/Notes) | lab notes |
| [`pythonlibrary`](pythonlibrary) | the installable Python package, its README, license and `pyproject.toml` |

## The Python library

`numericalanalysisiiser` collects the numerical methods from the labs. **Every function returns a pandas
DataFrame**, with one row per iteration, step size or evaluation point, so you can pick any columns and plot
them yourself. The optional `exact=` / `p_star=` argument adds `error` columns and the observed order of
convergence.

### Functions

| Area | Function | What it does |
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

## License

The library is released under the MIT license, see [pythonlibrary/LICENSE](pythonlibrary/LICENSE).

## Course content: MAT 308 Mathematics Lab [0 0 3 1]

The aim of the course is to familiarise students with the implementation of numerical analysis techniques,
especially numerical integration and solving linear systems of equations, through a set of weekly lab
assignments in Python.

| Lab | Topics | Library functions |
|---|---|---|
| 1 | Floating-point behaviour of `(e^x - 1) / x`; Lagrange and Newton interpolation | `lagrange`, `newton_interpolation`, `divided_differences` |
| 2 | Hermite interpolation and natural cubic splines | `hermite`, `natural_cubic_spline` |
| 3 | Numerical differentiation (forward, backward, central differences, Richardson extrapolation) and integration (trapezoidal, Simpson, composite rules, Romberg, Gauss-Legendre) | `finite_difference`, `richardson`, `trapezoid`, `simpson`, `romberg`, `gauss_legendre` |
| 4 | Root finding: bisection, fixed-point iteration, Newton's method, secant method | `bisection`, `fixed_point`, `newton`, `secant` |
| 5 | Jacobi and Gauss-Seidel methods | `jacobi`, `gauss_seidel` |
| 6 | Successive over-relaxation (SOR) | `sor` |
| 7 | Initial value problems: explicit Euler method, including a stiff problem | `euler` |
