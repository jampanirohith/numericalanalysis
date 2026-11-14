"""Numerical analysis methods that return pandas DataFrames.

    from numericalanalysis import bisection, richardson, jacobi, lagrange
"""

__version__ = "0.1.1"

from .lagrange import lagrange
from .newton_interpolation import newton_interpolation, divided_differences
from .hermite import hermite
from .natural_cubic_spline import natural_cubic_spline
from .finite_differences import finite_difference
from .richardson import richardson, richardson_triangle
from .trapezoid import trapezoid, composite_trapezoid
from .simpson import simpson, composite_simpson
from .romberg import romberg
from .gauss_legendre import gauss_legendre
from .bisection import bisection
from .fixed_point import fixed_point
from .newton import newton
from .secant import secant
from .jacobi import jacobi
from .gauss_seidel import gauss_seidel
from .sor import sor
from .tridiagonal_matrix import tridiagonal_matrix
from .euler import euler

__all__ = [
    "lagrange", "newton_interpolation", "divided_differences", "hermite", "natural_cubic_spline",
    "finite_difference", "richardson", "richardson_triangle",
    "trapezoid", "composite_trapezoid", "simpson", "composite_simpson", "romberg", "gauss_legendre",
    "bisection", "fixed_point", "newton", "secant",
    "jacobi", "gauss_seidel", "sor", "tridiagonal_matrix",
    "euler",
]
