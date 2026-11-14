"""Cross-check of numericalanalysisiiser against independent references (numpy / scipy)
for the problems of Labs 1-7.

    pip install numericalanalysisiiser scipy
    python crosscheck.py

Every line says OK or FAIL. "library" is the value computed by numericalanalysisiiser,
"reference" comes from numpy / scipy or from a closed form.
"""
import math
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import BarycentricInterpolator, CubicSpline, KroghInterpolator
from scipy.optimize import brentq

from numericalanalysisiiser import *

warnings.filterwarnings("ignore")          # scipy warns that 1e-14 is at round-off level

outcomes = []


def check(lab, what, library, reference, tol=1e-9):
    ok = abs(library - reference) <= tol * max(1.0, abs(reference))
    outcomes.append(ok)
    print(f"{'OK  ' if ok else 'FAIL'} {lab:<7}{what:<55}{library:>18.10g}{reference:>18.10g}")


def claim(lab, what, condition, shown=""):
    outcomes.append(bool(condition))
    print(f"{'OK  ' if condition else 'FAIL'} {lab:<7}{what:<55}{shown:>36}")


print(f"{'':<5} {'':<7}{'':<55}{'library':>18}{'reference':>18}")

# ------------------------------------------------------------------ Lab 1: interpolation
x, y = [-2, 0, -1, 1, 2], [4, 1, -1, 1, 9]
check("Lab 1", "Q2 Lagrange p(0.7), 5 nodes", lagrange(x, y, 0.7)["p"][0], np.polyval(np.polyfit(x, y, 4), 0.7))
check("Lab 1", "Q4 Newton p(-1) (the sheet says 12)", newton_interpolation([1, 2, 3, -4, 5], [2, 48, 272, 1182, 2262], -1)["p"][0], 12.0)

f = lambda t: 1 / (1 + t**2)
nodes = np.arange(-5, 6)
bary = BarycentricInterpolator(nodes, f(nodes))
q5 = newton_interpolation(list(nodes), list(f(nodes)), [4.2, 4.4, 4.6, 4.8], exact=f)
for a, e in zip(q5["alpha"], q5["error"]):
    check("Lab 1", f"Q5 error |f - p10| at x = {a:.1f}", e, abs(f(a) - bary(a)), tol=1e-7)

# ------------------------------------------------------------------ Lab 2: Hermite and splines
runge = lambda t: 1 / (1 + 25 * t**2)
drunge = lambda t: -50 * t / (1 + 25 * t**2) ** 2
P = np.linspace(-1, 1, 2001)
for N in (5, 10):
    xs = np.linspace(-1, 1, N)
    values = np.empty(2 * N)
    values[0::2], values[1::2] = runge(xs), drunge(xs)
    krogh = KroghInterpolator(np.repeat(xs, 2), values)
    mine = hermite(xs, runge(xs), drunge(xs), P, exact=runge)["error"].max()
    check("Lab 2", f"Q1 Hermite, N = {N}: max error", mine, np.max(np.abs(runge(P) - krogh(P))), tol=1e-7)

g2 = lambda t: np.exp(-t) * np.sin(2 * t)
fine = np.linspace(0, 4, 2001)
for h in (0.5, 0.25, 0.125):
    xs = np.arange(0, 4 + h / 2, h)
    mine = natural_cubic_spline(xs, g2(xs), fine, exact=g2)["error"].max()
    check("Lab 2", f"Q2 natural spline, h = {h}: max error", mine, np.max(np.abs(g2(fine) - CubicSpline(xs, g2(xs), bc_type="natural")(fine))))
fine = np.linspace(-1, 1, 2001)
for N in (10, 20, 50):
    xs = np.linspace(-1, 1, N)
    mine = natural_cubic_spline(xs, runge(xs), fine, exact=runge)["error"].max()
    check("Lab 2", f"Q3 natural spline, N = {N}: max error", mine, np.max(np.abs(runge(fine) - CubicSpline(xs, runge(xs), bc_type="natural")(fine))))

# ------------------------------------------------------------------ Lab 3: differentiation and integration
f1 = lambda t: math.exp(t) * math.cos(t)
d1 = math.exp(1) * (math.cos(1) - math.sin(1))
t = finite_difference(f1, 1.0, [2.0**-k for k in range(1, 11)], d1)
claim("Lab 3", "Q1 observed order: forward, backward, central", abs(t["forward_order"][3] - 1) < 0.1 and abs(t["backward_order"][3] - 1) < 0.1 and abs(t["central_order"][3] - 2) < 0.1,
      f"{t['forward_order'][3]:.2f} {t['backward_order'][3]:.2f} {t['central_order'][3]:.2f}")

f2 = lambda t: math.log(t**2 + 1) + math.sin(t**2)
d2 = 2 * 0.8 / (0.8**2 + 1) + 2 * 0.8 * math.cos(0.8**2)
t = finite_difference(f2, 0.8, [10.0**-k for k in range(1, 9)], d2, methods=("central",))
best = t["h"][t["central_error"].idxmin()]
claim("Lab 3", "Q2 error is smallest for a middle h, then grows (round-off)", 1e-7 <= best <= 1e-4 and t["central_error"].iloc[-1] > t["central_error"].min(), f"best h = {best:g}")

f3 = lambda t: math.exp(t) / (1 + t**2)
d3 = (math.exp(1) * 2 - math.exp(1) * 2) / 4
t = richardson(f3, 1.0, [2.0**-k for k in range(1, 7)], exact=d3)
claim("Lab 3", "Q3 order goes from 2 (D) to 4 (R)", abs(t["R0_order"][1] - 2) < 0.1 and abs(t["R1_order"][1] - 4) < 0.3, f"{t['R0_order'][1]:.2f} -> {t['R1_order'][1]:.2f}")

f4 = lambda t: math.exp(math.sin(t))
d4 = math.cos(math.pi / 6) * math.exp(0.5)
t = richardson(f4, math.pi / 6, 1.0, levels=5, exact=d4, layout="triangle")
check("Lab 3", "Q4 Richardson triangle, last diagonal entry", t["R5"][5], d4)

f5 = lambda t: math.sin(t) / t
d5 = math.sin(1) - 2 * math.cos(1)
t = richardson(f5, 1.0, [2.0**-k for k in range(1, 7)], method="second", exact=d5)
claim("Lab 3", "Q5 second derivative: order 2 (D2) and 4 (R)", abs(t["R0_order"][1] - 2) < 0.1 and abs(t["R1_order"][1] - 4) < 0.3, f"{t['R0_order'][1]:.2f} -> {t['R1_order'][1]:.2f}")

f6 = lambda t: math.exp(t) * math.sin(2 * t) + t**3 * math.log(t + 1)
d6 = (math.exp(0.5) * (-3 * math.sin(1) + 4 * math.cos(1)) + 6 * 0.5 * math.log(1.5) + 6 * 0.25 / 1.5 - 0.125 / 1.5**2)
t = richardson(f6, 0.5, [2.0**-k for k in range(1, 8)], levels=2, method="second", exact=d6)
claim("Lab 3", "Q6 orders about 2, 4, 6 (h = 1/4; finer h reaches round-off)", abs(t["R0_order"][1] - 2) < 0.1 and abs(t["R1_order"][1] - 4) < 0.1 and 5.0 < t["R2_order"][1] < 6.5,
      f"{t['R0_order'][1]:.2f} {t['R1_order'][1]:.2f} {t['R2_order'][1]:.2f}")

q7 = lambda t: math.sin(t) ** 2 - 2 * t * math.sin(t) + 1
ex7 = quad(q7, 0.75, 1.3, epsabs=1e-14)[0]
check("Lab 3", "Q7 trapezoid (1 interval) error", trapezoid(q7, 0.75, 1.3, exact=ex7)["error"][0], abs((1.3 - 0.75) / 2 * (q7(0.75) + q7(1.3)) - ex7))
check("Lab 3", "Q7 Simpson (2 sub-intervals) error", simpson(q7, 0.75, 1.3, exact=ex7)["error"][0], abs((1.3 - 0.75) / 6 * (q7(0.75) + 4 * q7(1.025) + q7(1.3)) - ex7))

ns = [4, 8, 16, 32, 64]
for number, fn, a, b in ((8, lambda t: math.exp(2 * t) * math.sin(3 * t), 0, 2), (9, lambda t: math.exp(t) * math.cos(2 * t) / (1 + t**2), 0, 1)):
    ex = quad(fn, a, b, epsabs=1e-14, epsrel=1e-14)[0]
    tt, ss = trapezoid(fn, a, b, ns, exact=ex), simpson(fn, a, b, ns, exact=ex)
    claim("Lab 3", f"Q{number} trapezoid order 2, Simpson order 4 (n = 4..64)", abs(tt["order"][2] - 2) < 0.1 and abs(ss["order"][2] - 4) < 0.2, f"{tt['order'][2]:.2f} {ss['order'][2]:.2f}")
    xs = np.linspace(a, b, 65)
    ys = np.array([fn(v) for v in xs])
    by_hand = (b - a) / 64 / 3 * (ys[0] + ys[-1] + 4 * ys[1:-1:2].sum() + 2 * ys[2:-1:2].sum())
    check("Lab 3", f"Q{number} Simpson n = 64 against a by-hand sum", ss["value"].iloc[-1], by_hand, tol=1e-12)

q10 = lambda t: math.exp(-t * t) * math.cos(2 * t)
check("Lab 3", "Q10 Romberg 6 rows, last diagonal entry", romberg(q10, 0, 1, levels=6)["R5"].iloc[-1], quad(q10, 0, 1, epsabs=1e-14)[0], tol=1e-9)
q11 = lambda t: t * t * math.exp(-t) / (1 + t * t)
r11 = romberg(q11, 0, 2, tol=1e-10)
last = len(r11) - 1
check("Lab 3", f"Q11 Romberg until diagonal change < 1e-10 ({len(r11)} rows)", r11[f"R{last}"][last], quad(q11, 0, 2, epsabs=1e-14)[0], tol=1e-9)

q12 = lambda t: math.exp(t) / (1 + t**2)
ex12 = quad(q12, 0, 2, epsabs=1e-14)[0]
by_hand2 = sum(w * q12(1 + s) for s, w in ((-1 / math.sqrt(3), 1.0), (1 / math.sqrt(3), 1.0)))
by_hand3 = sum(w * q12(1 + math.sqrt(3 / 5) * s) for s, w in ((-1, 5 / 9), (0, 8 / 9), (1, 5 / 9)))
t = gauss_legendre(q12, 0, 2, [2, 3], exact=ex12)
check("Lab 3", "Q12 Gauss-Legendre, 2 points (nodes from the sheet)", t["value"][0], by_hand2, tol=1e-12)
check("Lab 3", "Q12 Gauss-Legendre, 3 points (nodes from the sheet)", t["value"][1], by_hand3, tol=1e-12)

q13 = lambda t: math.exp(-t * t) * (1 + t * t + t**4)
t = gauss_legendre(q13, 0, 1, [2, 3, 4], exact=quad(q13, 0, 1, epsabs=1e-14)[0])
claim("Lab 3", "Q13 error decreases with the number of points", t["error"].is_monotonic_decreasing, " ".join(f"{e:.1e}" for e in t["error"]))

# ------------------------------------------------------------------ Lab 4: root finding
f = lambda t: t**3 - t - 2
root = brentq(f, 1, 2, xtol=1e-15)
t = bisection(f, 1, 2, tol=1e-10, p_star=root)
check("Lab 4", "Q1 bisection root", t["p"].iloc[-1], root, tol=1e-10)
claim("Lab 4", "Q1 iterations = ceil(log2((b-a)/tol)); error <= bound", len(t) == math.ceil(math.log2(1 / 1e-10)) and (t["error"] <= t["bound"] + 1e-15).all(), f"{len(t)} iterations")

f = lambda t: t**3 + t - 1
root = brentq(f, 0, 1, xtol=1e-15)
t1 = fixed_point(lambda t: math.cbrt(1 - t), 0.5, f=f, p_star=root)
t2 = fixed_point(lambda t: 1 - t**3, 0.5, f=f, p_star=root)
t3 = fixed_point(lambda t: (1 + t) / 2, 0.5, f=f, p_star=1.0)
claim("Lab 4", "Q2 g1 = (1-x)^(1/3) converges to the root", abs(t1["p"].iloc[-1] - root) < 1e-9, f"{len(t1) - 1} iterations")
claim("Lab 4", "Q2 g2 = 1-x^3 does not converge (|g'| > 1)", len(t2) == 1001, "1000 iterations, no convergence")
claim("Lab 4", "Q2 g3 = (1+x)/2 (as printed) converges to 1, not a root", abs(t3["p"].iloc[-1] - 1) < 1e-9 and abs(t3["f(p)"].iloc[-1]) > 0.5, f"residual {abs(t3['f(p)'].iloc[-1]):.2f}")

f = lambda t: math.exp(-t) - t
df = lambda t: -math.exp(-t) - 1
root = brentq(f, 0, 1, xtol=1e-15)
t = newton(f, df, 0.5, p_star=root)
check("Lab 4", "Q3 Newton root", t["p"].iloc[-1], root, tol=1e-12)
claim("Lab 4", "Q3 estimated order q is about 2", abs(t["q"].dropna().iloc[0] - 2) < 0.2, f"q = {t['q'].dropna().iloc[0]:.3f}")
claim("Lab 4", "Q3 starts 0.1 / 0.5 / 0.9 all converge", all(newton(f, df, p0)["step"].iloc[-2] < 1e-12 for p0 in (0.1, 0.5, 0.9)),
      "iterations " + " ".join(str(len(newton(f, df, p0)) - 1) for p0 in (0.1, 0.5, 0.9)))

f = lambda t: t * math.cos(t) - 1
s = secant(f, 0, 1, tol=1e-12)
check("Lab 4", "Q4 secant root (from p0 = 0, p1 = 1)", s["p"].iloc[-1], brentq(f, 4.8, 5.0, xtol=1e-15), tol=1e-11)

f = lambda t: t**3 - 2 * t - 5
root = brentq(f, 2, 3, xtol=1e-15)
runs = {"bisection": bisection(f, 2, 3, tol=1e-10, p_star=root), "fixed point": fixed_point(lambda t: (2 * t + 5) ** (1 / 3), 2.5, tol=1e-10, f=f, p_star=root),
        "newton": newton(f, lambda t: 3 * t**2 - 2, 2.5, tol=1e-10, p_star=root), "secant": secant(f, 2, 3, tol=1e-10, p_star=root)}
for name, tab in runs.items():
    check("Lab 4", f"Q5 {name}: root", tab["p"].iloc[-1], root, tol=1e-9)
iters = {n: len(tab) - (n != "bisection") for n, tab in runs.items()}
claim("Lab 4", "Q5 iterations: newton < secant < fixed point < bisection", iters["newton"] < iters["secant"] < iters["fixed point"] < iters["bisection"], " ".join(f"{v}" for v in iters.values()))

# ------------------------------------------------------------------ Lab 5 and 6: linear systems
A = [[10, -1, 2, 1], [-1, 11, -1, 3], [2, -1, 10, -1], [1, 3, -1, 8]]
b = [6, 25, -11, 15]
xs_ref = np.linalg.solve(A, b)
tj, tg = jacobi(A, b), gauss_seidel(A, b)
for name, tab in (("Jacobi", tj), ("Gauss-Seidel", tg)):
    sol = tab.filter(regex=r"^x\d+$").iloc[-1].to_numpy()
    check("Lab 5", f"Q1/2 {name}: max |x - np.linalg.solve|", np.max(np.abs(sol - xs_ref)), 0.0, tol=1e-5)
claim("Lab 5", "Q2 Gauss-Seidel needs fewer iterations than Jacobi", len(tg) < len(tj), f"{len(tj) - 1} vs {len(tg) - 1}")

A55, b55 = tridiagonal_matrix(55), np.ones(55)
tj, tg = jacobi(A55, b55, tol=1e-8, max_iter=500), gauss_seidel(A55, b55, tol=1e-8, max_iter=500)
D = np.diag(np.diag(A55))
rho_jacobi = max(abs(np.linalg.eigvals(np.eye(55) - np.linalg.solve(D, A55))))
check("Lab 5", "Q5 Jacobi asymptotic factor vs spectral radius", tj["q"].tail(5).mean(), rho_jacobi, tol=1e-2)
claim("Lab 5", "Q5 Gauss-Seidel faster than Jacobi (55 x 55)", len(tg) < len(tj), f"{len(tj) - 1} vs {len(tg) - 1}")
check("Lab 5", "Q3 Jacobi 55 x 55: max |x - np.linalg.solve|", np.max(np.abs(tj.filter(regex=r"^x\d+$").iloc[-1].to_numpy() - np.linalg.solve(A55, b55))), 0.0, tol=1e-6)

A6 = [[4, 1, -1, 1], [1, 4, -1, -1], [-1, -1, 5, 1], [1, -1, 1, 3]]
b6 = [-2, -1, 0, 1]
x6 = np.linalg.solve(A6, b6)
omegas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9]
table = pd.concat([sor(A6, b6, w, tol=1e-3) for w in omegas]).groupby("omega").last()
worst = max(np.max(np.abs(sor(A6, b6, w, tol=1e-12, max_iter=20000).filter(regex=r"^x\d+$").iloc[-1].to_numpy() - x6)) for w in omegas)
check("Lab 6", "Q1 SOR, all 17 omegas (tol 1e-12): worst |x - solve|", worst, 0.0, tol=1e-9)
claim("Lab 6", "Q1 all 17 omegas reach tol 1e-3", (table["step"] < 1e-3).all(), f"{len(table)} runs")
claim("Lab 6", "Q1 fewest iterations near omega = 1.1", table["k"].idxmin() in (1.1, 1.2), f"best omega {table['k'].idxmin()} ({table['k'].min()} its)")

# ------------------------------------------------------------------ Lab 7: ODEs
f = lambda t, y: -2 * y
t = euler(f, 0, 2, 1.0, [0.2, 0.1, 0.05], exact=lambda t: math.exp(-2 * t))
for h in (0.2, 0.1, 0.05):
    check("Lab 7", f"Q1 Euler y(2), h = {h}: library vs (1 - 2h)^N", t[t["h"] == h]["y"].iloc[-1], (1 - 2 * h) ** round(2 / h), tol=1e-12)
errors = t.groupby("h")["error"].max().sort_index(ascending=False)
claim("Lab 7", "Q1 E(h)/E(h/2) is about 2 (first order)", all(abs(errors.iloc[i] / errors.iloc[i + 1] - 2) < 0.3 for i in range(2)), " ".join(f"{errors.iloc[i] / errors.iloc[i + 1]:.3f}" for i in range(2)))

fs = lambda t, y: -1000 * (y - math.cos(t)) - math.sin(t)
t = euler(fs, 0, 1, 1.0, [0.01, 0.002, 0.001, 0.0005], exact=math.cos)
stable = t.groupby("h", sort=False)["stable"].all().tolist()
claim("Lab 7", "Q2 stable only for h <= 0.002 (|1 - 1000h| <= 1)", stable == [False, True, True, True], str(stable))

print()
print(f"{sum(outcomes)} of {len(outcomes)} checks passed")
