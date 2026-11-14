# Cross-checking the lab solutions with `numericalanalysisiiser`

This guide is for anyone who wants to **install the library, run the lab problems, and check the
results independently**. The methods from the Labs 1 to 7 of MAT 308 (Mathematics Lab) are collected in the Python
package [`numericalanalysisiiser`](https://pypi.org/project/numericalanalysisiiser/). Every method is a
short function in its own file, so the implementation can be read and compared with the textbook algorithm.

## 1. Install

You need Python 3.9 or newer. Optionally use a fresh virtual environment so nothing else is affected.

Windows (PowerShell):

```powershell
python -m venv crosscheck-env
crosscheck-env\Scripts\python -m pip install numericalanalysisiiser scipy
```

macOS / Linux:

```bash
python3 -m venv crosscheck-env
crosscheck-env/bin/python -m pip install numericalanalysisiiser scipy
```

This installs the library with numpy and pandas. `scipy` is only used by the cross-check script as an
independent reference (`quad`, `brentq`, `CubicSpline`, ...); the library itself does not need it.

## 2. Run the cross-check

Download [`crosscheck.py`](crosscheck.py) from this repository and run it with the environment's Python:

```powershell
crosscheck-env\Scripts\python crosscheck.py
```

(macOS / Linux: `crosscheck-env/bin/python crosscheck.py`.)

It runs the problems of the Lab 1 to Lab 7 sheets through the library and compares each result with an
independent computation: numpy / scipy, a closed form, or the value printed on the sheet. Each line
says `OK` or `FAIL`. The last line is the total. With the published version it prints
**59 of 59 checks passed**.

The comparisons are of two kinds:

* **Value checks**: library value and reference value side by side, for example the Newton interpolation
  `p(-1) = 12`, the spline errors against `scipy.interpolate.CubicSpline`, the Romberg value against
  `scipy.integrate.quad`, the Jacobi / Gauss-Seidel solutions against `numpy.linalg.solve`.
* **Behaviour checks**: the property the question asks about, for example the observed order of
  convergence (forward and backward differences 1, central 2, Richardson 4, Newton's method 2, Euler 1),
  that `g2(x) = 1 - x^3` does not converge, or that explicit Euler is unstable for `h = 0.01` and stable
  for `h <= 0.002` on the stiff problem.

<details>
<summary>Output of the run</summary>

```
                                                                               library         reference
OK   Lab 1  Q2 Lagrange p(0.7), 5 nodes                                     0.9824125         0.9824125
OK   Lab 1  Q4 Newton p(-1) (the sheet says 12)                                    12                12
OK   Lab 1  Q5 error |f - p10| at x = 4.2                                0.5067963759      0.5067963759
OK   Lab 1  Q5 error |f - p10| at x = 4.4                                 1.191097572       1.191097572
OK   Lab 1  Q5 error |f - p10| at x = 4.6                                 1.800719013       1.800719013
OK   Lab 1  Q5 error |f - p10| at x = 4.8                                 1.762788118       1.762788118
OK   Lab 2  Q1 Hermite, N = 5: max error                                 0.2235785854      0.2235785854
OK   Lab 2  Q1 Hermite, N = 10: max error                                 2.037326959       2.037326959
OK   Lab 2  Q2 natural spline, h = 0.5: max error                       0.05682726704     0.05682726704
OK   Lab 2  Q2 natural spline, h = 0.25: max error                      0.01282167234     0.01282167234
OK   Lab 2  Q2 natural spline, h = 0.125: max error                    0.003104084026    0.003104084026
OK   Lab 2  Q3 natural spline, N = 10: max error                         0.1428743541      0.1428743541
OK   Lab 2  Q3 natural spline, N = 20: max error                        0.01233557944     0.01233557944
OK   Lab 2  Q3 natural spline, N = 50: max error                      0.0001484448161   0.0001484448161
OK   Lab 3  Q1 observed order: forward, backward, central                                1.02 0.98 2.00
OK   Lab 3  Q2 error is smallest for a middle h, then grows (round-off)                      best h = 1e-06
OK   Lab 3  Q3 order goes from 2 (D) to 4 (R)                                              2.02 -> 3.93
OK   Lab 3  Q4 Richardson triangle, last diagonal entry                   1.427834504       1.427834504
OK   Lab 3  Q5 second derivative: order 2 (D2) and 4 (R)                                   2.00 -> 4.00
OK   Lab 3  Q6 orders about 2, 4, 6 (h = 1/4; finer h reaches round-off)                      1.98 4.00 5.61
OK   Lab 3  Q7 trapezoid (1 interval) error                             0.01664745675     0.01664745675
OK   Lab 3  Q7 Simpson (2 sub-intervals) error                        0.0001052060686   0.0001052060686
OK   Lab 3  Q8 trapezoid order 2, Simpson order 4 (n = 4..64)                                 2.00 4.05
OK   Lab 3  Q8 Simpson n = 64 against a by-hand sum                      -14.21397094      -14.21397094
OK   Lab 3  Q9 trapezoid order 2, Simpson order 4 (n = 4..64)                                 2.00 4.01
OK   Lab 3  Q9 Simpson n = 64 against a by-hand sum                      0.5397877185      0.5397877185
OK   Lab 3  Q10 Romberg 6 rows, last diagonal entry                      0.4290977821      0.4290977821
OK   Lab 3  Q11 Romberg until diagonal change < 1e-10 (8 rows)           0.2593386854      0.2593386854
OK   Lab 3  Q12 Gauss-Legendre, 2 points (nodes from the sheet)           2.682925906       2.682925906
OK   Lab 3  Q12 Gauss-Legendre, 3 points (nodes from the sheet)           2.660186904       2.660186904
OK   Lab 3  Q13 error decreases with the number of points                       3.2e-03 1.8e-04 6.2e-06
OK   Lab 4  Q1 bisection root                                             1.521379707       1.521379707
OK   Lab 4  Q1 iterations = ceil(log2((b-a)/tol)); error <= bound                         34 iterations
OK   Lab 4  Q2 g1 = (1-x)^(1/3) converges to the root                                     67 iterations
OK   Lab 4  Q2 g2 = 1-x^3 does not converge (|g'| > 1)                  1000 iterations, no convergence
OK   Lab 4  Q2 g3 = (1+x)/2 (as printed) converges to 1, not a root                       residual 1.00
OK   Lab 4  Q3 Newton root                                               0.5671432904      0.5671432904
OK   Lab 4  Q3 estimated order q is about 2                                                   q = 2.005
OK   Lab 4  Q3 starts 0.1 / 0.5 / 0.9 all converge                                     iterations 5 4 5
OK   Lab 4  Q4 secant root (from p0 = 0, p1 = 1)                          4.917185925       4.917185925
OK   Lab 4  Q5 bisection: root                                            2.094551482       2.094551482
OK   Lab 4  Q5 fixed point: root                                          2.094551482       2.094551482
OK   Lab 4  Q5 newton: root                                               2.094551482       2.094551482
OK   Lab 4  Q5 secant: root                                               2.094551482       2.094551482
OK   Lab 4  Q5 iterations: newton < secant < fixed point < bisection                           34 13 5 7
OK   Lab 5  Q1/2 Jacobi: max |x - np.linalg.solve|                     3.25974581e-07                 0
OK   Lab 5  Q1/2 Gauss-Seidel: max |x - np.linalg.solve|              1.101361137e-07                 0
OK   Lab 5  Q2 Gauss-Seidel needs fewer iterations than Jacobi                                  17 vs 9
OK   Lab 5  Q5 Jacobi asymptotic factor vs spectral radius                        0.5      0.4992134075
OK   Lab 5  Q5 Gauss-Seidel faster than Jacobi (55 x 55)                                       26 vs 17
OK   Lab 5  Q3 Jacobi 55 x 55: max |x - np.linalg.solve|              7.450580486e-09                 0
OK   Lab 6  Q1 SOR, all 17 omegas (tol 1e-12): worst |x - solve|      2.297784185e-11                 0
OK   Lab 6  Q1 all 17 omegas reach tol 1e-3                                                     17 runs
OK   Lab 6  Q1 fewest iterations near omega = 1.1                                best omega 1.1 (6 its)
OK   Lab 7  Q1 Euler y(2), h = 0.2: library vs (1 - 2h)^N                0.0060466176      0.0060466176
OK   Lab 7  Q1 Euler y(2), h = 0.1: library vs (1 - 2h)^N               0.01152921505     0.01152921505
OK   Lab 7  Q1 Euler y(2), h = 0.05: library vs (1 - 2h)^N              0.01478088294     0.01478088294
OK   Lab 7  Q1 E(h)/E(h/2) is about 2 (first order)                                         2.222 2.094
OK   Lab 7  Q2 stable only for h <= 0.002 (|1 - 1000h| <= 1)                  [False, True, True, True]

59 of 59 checks passed
```

</details>

## 3. Look at one problem yourself

Every method returns a pandas DataFrame, so the whole table (not just the final number) is there to
inspect, and the last row is the answer for the iterative methods.

```python
import math
from numericalanalysisiiser import bisection, newton, jacobi, richardson, trapezoid, simpson, euler

# Lab 4, Q1: bisection for x^3 - x - 2 on [1, 2], tolerance 1e-10
t = bisection(lambda x: x**3 - x - 2, 1, 2, tol=1e-10, p_star=1.5213797068045676)
print(t[["n", "a", "b", "p", "f(p)", "b-a", "bound", "error"]])
print(len(t), "iterations; root", t["p"].iloc[-1])

# Lab 3, Q8: composite trapezoid and Simpson, n = 4 ... 64
f = lambda x: math.exp(2 * x) * math.sin(3 * x)
F = lambda x: math.exp(2 * x) * (2 * math.sin(3 * x) - 3 * math.cos(3 * x)) / 13
exact = F(2) - F(0)
print(trapezoid(f, 0, 2, [4, 8, 16, 32, 64], exact=exact))
print(simpson(f, 0, 2, [4, 8, 16, 32, 64], exact=exact))

# Lab 5: Jacobi on the 4 x 4 system
A = [[10, -1, 2, 1], [-1, 11, -1, 3], [2, -1, 10, -1], [1, 3, -1, 8]]
b = [6, 25, -11, 15]
print(jacobi(A, b))
```

| Lab | Problem | Call |
|---|---|---|
| 1 | Lagrange / Newton interpolation | `lagrange(x, y, alpha)`, `newton_interpolation(x, y, alpha)`, `divided_differences(x, y)` |
| 2 | Hermite, natural cubic spline | `hermite(x, y, dy, alpha, exact=f)`, `natural_cubic_spline(x, y, alpha, exact=f)` |
| 3 | Differences, Richardson | `finite_difference(f, x, h, exact)`, `richardson(f, x, h, levels, method, exact)` |
| 3 | Integration | `trapezoid`, `simpson`, `romberg`, `gauss_legendre` |
| 4 | Root finding | `bisection`, `fixed_point`, `newton`, `secant` |
| 5, 6 | Jacobi, Gauss-Seidel, SOR | `jacobi(A, b)`, `gauss_seidel(A, b)`, `sor(A, b, omega)` |
| 7 | Euler method | `euler(f, t0, T, y0, h, exact)` |

The full description of every function, with its arguments, the columns it returns and worked examples,
is in [pythonlibrary/README.md](pythonlibrary/README.md).

## 4. Where the code is

One file per method in [`pythonlibrary/numericalanalysisiiser`](pythonlibrary/numericalanalysisiiser):
for example `bisection.py`, `newton.py`, `secant.py`, `richardson.py`, `romberg.py`, `jacobi.py`, `sor.py`,
`euler.py`. Each file contains only that method and is a few dozen lines of plain Python, written in the style of the
lab scripts in [`College/Lab`](College/Lab).

## 5. Notes on how the sheets were read

* **Lab 2 Q3 and Q1:** "N equally spaced points" is read as `N` nodes (`np.linspace(-1, 1, N)`), as in the
  formula `x_i = -1 + 2i/(N-1)` of Q1. Hermite interpolation with many equally spaced nodes (N = 20, 50, 300)
  is numerically unstable (Runge phenomenon), so the cross-check compares only `N = 5` and `N = 10`.
* **Lab 4 Q2:** `g3(x) = (1 + x)/2` is used as printed. Its fixed point is `1`, which is not a root of
  `x^3 + x - 1`, so the iteration converges but the residual `|f(p)|` stays at 1.
* **Lab 4 Q4:** starting from `p0 = 0`, `p1 = 1`, the secant method for `x cos x - 1` converges to the root
  `4.917185925...`. The function has no root in `[0, 1]`, since `x cos x < 1` there.
* **Lab 5:** the 4 x 4 matrix is the one printed on the sheet.
* **Stopping rules:** Jacobi, Gauss-Seidel and SOR stop when `max|x_k - x_(k-1)| < tol`, as the sheets say.
  For small `omega` the error can then be larger than `tol`, because the iterates change slowly.
* **Not covered:** Lab 1 Q1 and Q3 (a plain loop and a plot), and the later topics of the course (LU
  decomposition, Runge-Kutta methods, boundary value problems, the power method) are not in the library yet.
