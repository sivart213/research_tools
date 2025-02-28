# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

import re

from typing import Union, Optional, Tuple, List

import numpy as np
import sympy as sp
import scipy.special as scs

from sklearn import metrics
from scipy import stats, optimize, linalg

from ..functions import get_const, has_units, extract_variable, ode_bounds


# %% General
def sind(angle: float) -> float:
    """
    Return the sine of the angle in degrees.
    EQ: sin(angle) = sin(angle * π / 180)
    
    Args:
        angle (float): Angle in degrees.
    
    Returns:
        float: Sine of the angle.
    """
    return np.sin(np.radians(angle))


def cosd(angle: float) -> float:
    """
    Return the cosine of the angle in degrees.
    EQ: cos(angle) = cos(angle * π / 180)
    
    Args:
        angle (float): Angle in degrees.
    
    Returns:
        float: Cosine of the angle.
    """
    return np.cos(np.radians(angle))


def tand(angle: float) -> float:
    """
    Return the tangent of the angle in degrees.
    EQ: tan(angle) = tan(angle * π / 180)
    
    Args:
        angle (float): Angle in degrees.
    
    Returns:
        float: Tangent of the angle.
    """
    return np.tan(np.radians(angle))


def arcsind(x: float) -> float:
    """
    Return the arcsine in degrees.
    EQ: arcsin(x) = arcsin(x) * 180 / π
    
    Args:
        x (float): Value.
    
    Returns:
        float: Arcsine in degrees.
    """
    return np.degrees(np.arcsin(x))


def arccosd(x: float) -> float:
    """
    Return the arccosine in degrees.
    EQ: arccos(x) = arccos(x) * 180 / π
    
    Args:
        x (float): Value.
    
    Returns:
        float: Arccosine in degrees.
    """
    return np.degrees(np.arccos(x))


def inv_sum_invs(*arr: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the inverse of the sum of inverses.
    EQ: 1 / sum(1 / arr)
    
    Args:
        *arr: List of values.
    
    Returns:
        float: Inverse of the sum of inverses.
    """
    if isinstance(arr[0], (list, np.ndarray)):
        arr = arr[0]
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return 1 / sum(1 / arr)


def arrh(A0: float, Ea: float, T: float) -> float:
    """
    Calculate the Arrhenius equation.
    EQ: A = A0 * exp(-Ea / (k_B * T))
    
    Args:
        A0 (float): Pre-exponential factor.
        Ea (float): Activation energy (eV).
        T (float): Temperature (K).
    
    Returns:
        float: Arrhenius value.
    """
    w_units = has_units(vars())

    k_B = get_const("boltzmann", w_units, ["eV", "K"])

    res = A0 * sp.exp(-1 * Ea / (k_B * T))

    if isinstance(res, sp.Number):
        return float(res)
    return res


def erf(z: float) -> float:
    """
    Calculate the error function.
    EQ: erf(z) = 2 / sqrt(π) * ∫ exp(-t^2) dt from 0 to z
    
    Args:
        z (float): Value.
    
    Returns:
        float: Error function value.
    """
    if isinstance(z, sp.Basic):
        res = sp.erf(z)
    else:
        res = scs.erf(z)

    if isinstance(res, sp.Number):
        return float(res)
    return res


def erfinv(z: float) -> float:
    """
    Calculate the inverse error function.
    EQ: erfinv(z) = inverse of erf(z)
    
    Args:
        z (float): Value.
    
    Returns:
        float: Inverse error function value.
    """
    if isinstance(z, sp.Basic):
        res = sp.erfinv(z)
    else:
        res = scs.erfinv(z) # https://github.com/PyCQA/pylint/issues/3744 pylint: disable=no-member

    if isinstance(res, sp.Number):
        return float(res)
    return res


def erfc(z: float) -> float:
    """
    Calculate the complementary error function.
    EQ: erfc(z) = 1 - erf(z)
    
    Args:
        z (float): Value.
    
    Returns:
        float: Complementary error function value.
    """
    if isinstance(z, sp.Basic):
        res = sp.erfc(z)
    else:
        res = scs.erfc(z)

    if isinstance(res, sp.Number):
        return float(res)
    return res


def erfcinv(z: float) -> float:
    """
    Calculate the inverse complementary error function.
    EQ: erfcinv(z) = inverse of erfc(z)
    
    Args:
        z (float): Value.
    
    Returns:
        float: Inverse complementary error function value.
    """
    if isinstance(z, sp.Basic):
        res = sp.erfcinv(z)
    else:
        res = scs.erfcinv(z)

    if isinstance(res, sp.Number):
        return float(res)
    return res


def polynomial(*coeff: float, x: Optional[float] = None) -> float:
    """
    Compute polynomial using Horner's Method.
    EQ: P(x) = C0 + C1*x + C2*x^2 + ... + Cn*x^n
    
    Args:
        *coeff: Coefficients of the polynomial.
        x (float, optional): Variable. Default is None.
    
    Returns:
        float: Polynomial value.
    
    Notes:
        C is a vector of coefficients, highest order coefficient at C[0].
    """
    if x is None:
        x = sp.symbols("x", real=True)
    res = 0
    for Cn in coeff:
        res = x * res + Cn

    if isinstance(res, sp.Number):
        return float(res)
    elif hasattr(res, "expand"):
        return res.expand()
    return res

def piecewise(*args: Union[Tuple[float, float], List[float]], var: str = "x", **kwargs) -> sp.Piecewise:
    """
    Compute piecewise function.
    EQ: Piecewise((expr1, cond1), (expr2, cond2), ...)
    
    Args:
        *args: List of tuples representing piecewise conditions.
        var (str, optional): Variable. Default is "x".
        **kwargs: Additional arguments.
    
    Returns:
        sympy.Piecewise: Piecewise function.
    """
    args=tuple(args)
    if not isinstance(args[0], (list, tuple)):
        args = tuple([args])
    elif isinstance(args[0][0], (list, tuple)):
        args = tuple(args[0])

    if len(args) == 1 and len(args[0]) == 1:
        return args[0][0]

    if not isinstance(args[-1], (tuple, list)):
        args = args[:-1] + tuple([(args[-1], True)])
    elif len(args[-1]) == 1:
        # args[-1] = (args[-1][0], True)
        args = args[:-1] + tuple([(args[-1][0], True)])
    elif not isinstance(args[-1][-1], bool):
        args = args+tuple([(0, True)])

    if isinstance(var, str):
        var = sp.Symbol(var, real=True)
    pairs = []
    for a in args:
        if len(a) > 2:
            pairs.append((a[0], sp.Interval(*a[1:]).contains(var)))
        elif isinstance(a[1], (tuple, list)):
            pairs.append((a[0], sp.Interval(*a[1]).contains(var)))
        elif isinstance(a[1], bool):
            pairs.append(a)
        elif not isinstance(a[1], str):
            pairs.append((a[0], var < a[1]))
        else:
            a1 = re.search("[<>=]+", a[1])
            var1 = re.search(str(var), a[1])
            kvars = kwargs.get("kwargs", kwargs)
            kvars[str(var)] = var
            if not var1:
                if not a1:
                    expr = sp.parse_expr(str(var)+"<"+a[1])
                elif a1.start() == 0:
                    expr = sp.parse_expr(str(var)+a[1])
                elif a1.end() == len(a[1]):
                    expr = sp.parse_expr(a[1]+str(var))
                else:
                    expr = sp.parse_expr(str(var)+"*"+a[1])
            else:
                expr = sp.parse_expr(a[1])
            pairs.append((a[0], expr.subs(kvars)))

    return sp.Piecewise(*[(a[0], a[1]) for a in pairs], evaluate=False)

def ode(f: Optional[sp.Function] = None, x: Optional[sp.Symbol] = None, deg: int = 2, expr: Union[sp.Expr, int] = 0, **kwargs) -> sp.Eq:
    """
    Compute ordinary differential equation.
    EQ: d^deg(f(x)) / dx^deg = expr
    
    Args:
        f (sympy.Function, optional): Function. Default is None.
        x (sympy.Symbol, optional): Variable. Default is None.
        deg (int, optional): Degree of the differential equation. Default is 2.
        expr (sympy.Expr, optional): Expression. Default is 0.
        **kwargs: Additional arguments.
    
    Returns:
        sympy.Eq: Differential equation.
    """
    if isinstance(expr, sp.Piecewise):
        expr = sp.piecewise_fold(expr)

    if f is None:
        f = sp.symbols("f", cls=sp.Function)
    if x is None:
        x = [extract_variable(expr, "x")]
        x = x[0] if len(x) >= 1 and x[0] is not None else sp.Symbol("x", real=True)
    
    res = sp.Eq(f(x).diff(*[x]*deg), expr)
    if kwargs.get("solve", False):
        bnds = kwargs.get("bnds", kwargs.get("bounds"))
        if bnds is None:
            res = sp.dsolve(res, f(x), simplify=False)
        else:
            res = sp.dsolve(res, f(x), simplify=False, ics=ode_bounds(f, x, bnds=bnds))
        if not kwargs.get("full", False):
            res = res.rhs
        if kwargs.get("simplify", True):
            res = res.simplify()
    if kwargs.get("all", False):
        return res, f, x
    return res



def integral(x: Optional[sp.Symbol] = None, bound: Optional[Tuple[float, float]] = None, deg: int = 2, expr: Union[sp.Expr, int] = 0, **kwargs) -> Union[sp.Expr, Tuple[sp.Expr, sp.Symbol]]:
    """
    Compute integral.
    EQ: ∫ expr dx
    
    Args:
        x (sympy.Symbol, optional): Variable. Default is None.
        bound (tuple, optional): Integration bounds. Default is None.
        deg (int, optional): Degree of the integral. Default is 2.
        expr (sympy.Expr, optional): Expression. Default is 0.
        **kwargs: Additional arguments.
    
    Returns:
        sympy.Expr: Integral.
    """
    if isinstance(expr, sp.Piecewise):
        expr = sp.piecewise_fold(expr)

    if x is None:
        x = [extract_variable(expr, "x")]
        x = x[0] if len(x) >= 1 else sp.Symbol("x", real=True)
    
    res = expr
    for n in range(deg):
        if isinstance(bound, (tuple, list, np.ndarray)):
            if isinstance(bound[n], (tuple, list, np.ndarray)):
                res = sp.integrate(res, (x, bound[n][0], bound[n][1]))
            else:
                res = sp.integrate(res, (x, bound[0], bound[1]))
        else:
            if kwargs.get("constants", kwargs.get("const", False)):
                res = sp.integrate(res, (x)) + sp.Symbol(f"C_{n}", real=True)
            else:
                res = sp.integrate(res, (x))
    if kwargs.get("simplify", True):
        res = res.simplify()
    if kwargs.get("all", False):
        return res, x
    return res



# %% Geometric
def line(x: float, m: float = 1, b: float = 0) -> float:
    """
    Calculate the equation of a line.
    EQ: y = m * x + b
    
    Args:
        x (float): Variable.
        m (float, optional): Slope. Default is 1.
        b (float, optional): Intercept. Default is 0.
    
    Returns:
        float: Line equation value.
    """
    res = m * x + b
    return res


def arc(h: float, a: float, r: float) -> float:
    """
    Calculate the radius of an arc.
    EQ: r = (a^2 + h^2) / (2 * h)
    
    Args:
        h (float): Height.
        a (float): Chord length.
        r (float): Radius.
    
    Returns:
        float: Radius of the arc.
    """
    if isinstance(a, (list, tuple, np.ndarray)):
        a = sp.sqrt(a[0] ** 2 + a[1] ** 2)
    res = (a**2 + h**2) / (2 * h)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def sphere_vol(h: Optional[float] = None, a: Optional[float] = None, r: Optional[float] = None) -> float:
    """
    Calculate the volume of a sphere.
    EQ: V = π * h^2 / 3 * (3 * r - h)
    
    Args:
        h (float, optional): Height. Default is None.
        a (float, optional): Chord length. Default is None.
        r (float, optional): Radius. Default is None.
    
    Returns:
        float: Volume of the sphere.
    """
    if isinstance(a, (list, tuple, np.ndarray)):
        a = sp.sqrt(a[0] ** 2 + a[1] ** 2)

    if h is None and a is None:
        h = 2 * r
    elif r is None:
        r = arc(h, a, None)
    elif h is None:
        h = arc(None, a, r)

    res = np.pi * h**2 / 3 * (3 * r - h)
    if isinstance(res, sp.Number) or isinstance(res/sp.pi, sp.Number):
        return float(res)
    return res


def sphere_area(h: Optional[float] = None, a: Optional[float] = None, r: Optional[float] = None) -> float:
    """
    Calculate the surface area of a sphere.
    EQ: A = 2 * π * r * h
    
    Args:
        h (float, optional): Height. Default is None.
        a (float, optional): Chord length. Default is None.
        r (float, optional): Radius. Default is None.
    
    Returns:
        float: Surface area of the sphere.
    """
    if isinstance(a, (list, tuple, np.ndarray)):
        a = sp.sqrt(a[0] ** 2 + a[1] ** 2)

    if h is None and a is None:
        h = 2 * r
    elif r is None:
        r = arc(h, a, None)
    elif h is None:
        h = arc(None, a, r)

    res = 2 * sp.pi * r * h
    if isinstance(res, sp.Number) or isinstance(res/sp.pi, sp.Number):
        return float(res)
    return res


# %% Temperature


def FtoC(T: float) -> float:
    """
    Convert Fahrenheit to Celsius.
    EQ: T_C = (T_F - 32) * 5 / 9
    
    Args:
        T (float): Temperature in Fahrenheit.
    
    Returns:
        float: Temperature in Celsius.
    """
    res = (T - 32) * 5.0 / 9.0
    return res


def FtoK(T: float) -> float:
    """
    Convert Fahrenheit to Kelvin.
    EQ: T_K = (T_F - 32) * 5 / 9 + 273.15
    
    Args:
        T (float): Temperature in Fahrenheit.
    
    Returns:
        float: Temperature in Kelvin.
    """
    res = (T - 32) * 5.0 / 9.0 + 273.15
    return res


def FtoR(T: float) -> float:
    """
    Convert Fahrenheit to Rankine.
    EQ: T_R = T_F + 459.67
    
    Args:
        T (float): Temperature in Fahrenheit.
    
    Returns:
        float: Temperature in Rankine.
    """
    res = T - 32 + (273.15 * 9.0 / 5.0)
    return res


def CtoF(T: float) -> float:
    """
    Convert Celsius to Fahrenheit.
    EQ: T_F = T_C * 9 / 5 + 32
    
    Args:
        T (float): Temperature in Celsius.
    
    Returns:
        float: Temperature in Fahrenheit.
    """
    res = T * 9.0 / 5.0 + 32
    return res


def CtoK(T: float) -> float:
    """
    Convert Celsius to Kelvin.
    EQ: T_K = T_C + 273.15
    
    Args:
        T (float): Temperature in Celsius.
    
    Returns:
        float: Temperature in Kelvin.
    """
    res = T + 273.15
    return res


def CtoR(T: float) -> float:
    """
    Convert Celsius to Rankine.
    EQ: T_R = (T_C + 273.15) * 9 / 5
    
    Args:
        T (float): Temperature in Celsius.
    
    Returns:
        float: Temperature in Rankine.
    """
    res = (T + 273.15) * 9.0 / 5.0
    return res


def KtoF(T: float) -> float:
    """
    Convert Kelvin to Fahrenheit.
    EQ: T_F = (T_K - 273.15) * 9 / 5 + 32
    
    Args:
        T (float): Temperature in Kelvin.
    
    Returns:
        float: Temperature in Fahrenheit.
    """
    res = (T - 273.15) * 9.0 / 5.0 + 32
    return res


def KtoC(T: float) -> float:
    """
    Convert Kelvin to Celsius.
    EQ: T_C = T_K - 273.15
    
    Args:
        T (float): Temperature in Kelvin.
    
    Returns:
        float: Temperature in Celsius.
    """
    res = T - 273.15
    return res


def KtoR(T: float) -> float:
    """
    Convert Kelvin to Rankine.
    EQ: T_R = T_K * 9 / 5
    
    Args:
        T (float): Temperature in Kelvin.
    
    Returns:
        float: Temperature in Rankine.
    """
    res = T * 9.0 / 5.0
    return res


def RtoF(T: float) -> float:
    """
    Convert Rankine to Fahrenheit.
    EQ: T_F = T_R - 459.67
    
    Args:
        T (float): Temperature in Rankine.
    
    Returns:
        float: Temperature in Fahrenheit.
    """
    res = T + 32 - (273.15 * 9.0 / 5.0)
    return res


def RtoC(T: float) -> float:
    """
    Convert Rankine to Celsius.
    EQ: T_C = (T_R - 491.67) * 5 / 9
    
    Args:
        T (float): Temperature in Rankine.
    
    Returns:
        float: Temperature in Celsius.
    """
    res = (T * 5.0 / 9.0) - 273.15
    return res


def RtoK(T: float) -> float:
    """
    Convert Rankine to Kelvin.
    EQ: T_K = T_R * 5 / 9
    
    Args:
        T (float): Temperature in Rankine.
    
    Returns:
        float: Temperature in Kelvin.
    """
    res = T * 5.0 / 9.0
    return res


# %% Statistics and costs
class Statistics:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(
        self,
        true,
        pred,
        weight=None,
        indep=None,
        n_param=1,
        resid_type="",
    ):
        if not isinstance(true, np.ndarray):
            true = np.array(true)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)

        self.true = true
        self.pred = pred
        self.indep = indep
        self.weight = weight
        self.n_param = n_param
    

        if "int" in resid_type.lower():
            self._resid = self.int_std_res
        elif "ext" in resid_type.lower():
            self._resid = self.ext_std_res
        else:
            self._resid = self.true - self.pred

    def __getitem__(self, name):
        """Get item, defined with shorthand options."""
        shorthand = {
            "mape": "mean_abs_perc_err",
            "msle": "mean_sq_log_err",
            "rmse": "root_mean_sq_err",
            "mse": "mean_sq_err",
            "rmae": "root_mean_abs_err",
            "mae": "mean_abs_err",
            "medae": "median_abs_err",
            "r2": "r_sq",
            "var": "explained_var_score",
        }

        if name.lower() in shorthand.keys():
            return getattr(self, shorthand[name.lower()])
        else:
            return getattr(self, name)

    @property
    def indep(self):
        """Return SIMS data in log or normal form."""
        return self._indep

    @indep.setter
    def indep(self, value):
        """Set SIMS data."""
        if value is None:
            self._indep = np.arange(0, len(self.true))
        else:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            self._indep = value

    @property
    def dft(self):
        """Return degrees of freedom population dep. variable variance."""
        return self.indep.shape[0] - self.n_param

    @property
    def df(self):
        """Return degrees of freedom."""
        return self.dft

    @property
    def dfe(self):
        """Return degrees of freedom population error variance."""
        return self.indep.shape[0]

    @property
    def residuals(self):
        """Return calculated external standardized residual.."""
        return self.true - self.pred

    @property
    def sq_err(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.square(self.residuals)

    @property
    def sse(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.sum(self.sq_err)

    @property
    def sst(self):
        """Return total sum of squared errors (actual vs avg(actual))."""
        return np.sum(
            (self.true - np.average(self.true, weights=self.weight, axis=0)) ** 2
        )

    @property
    def r_sq(self):
        """Return calculated r2_score."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.r2_score(self.true, self.pred, sample_weight=self.weight)
        else:
            return 1 - self.sse / self.sst

    @property
    def adj_r_sq(self):
        """Return calculated value of adjusted r^2."""
        return 1 - (self.sse / self.dfe) / (self.sst / self.dft)

    @property
    def int_std_res(self):
        """Return calculated internal standardized residual."""
        n = len(self.indep)
        diff_mean_sqr = np.dot(
            (self.indep - np.average(self.indep, weights=self.weight, axis=0)),
            (self.indep - np.average(self.indep, weights=self.weight, axis=0)),
        )
        h_ii = (
            self.indep - np.average(self.indep, weights=self.weight, axis=0)
        ) ** 2 / diff_mean_sqr + (1 / n)
        Var_e = np.sqrt(sum((self.residuals) ** 2) / (n - 2))
        return (self.residuals) / (Var_e * ((1 - h_ii) ** 0.5))

    @property
    def ext_std_res(self):
        """Return calculated external standardized residual.."""
        r = self.int_std_res
        n = len(r)
        return [r_i * np.sqrt((n - 2 - 1) / (n - 2 - r_i**2)) for r_i in r]

    @property
    def normal_test(self):
        """Return calculated value from ks."""
        try:
            self.stat, self.p = stats.normaltest(self._resid)
        except (ValueError, TypeError):
            self.p = np.nan
            self.stat = np.nan
        return self.p

    @property
    def shap_test(self):
        """Return calculated value of the shap test."""
        try:
            self.shap_stat, self.shap_p = stats.shapiro(self._resid)
        except (ValueError, TypeError):
            self.shap_p = np.nan
            self.shap_stat = np.nan
        return self.shap_p

    @property
    def ks_test(self):
        """Return calculated value from ks."""
        self.ks_stat, self.ks_p = stats.ks_2samp(self.pred, self.true)
        return self.ks_p

    @property
    def chi_sq(self):
        """Return calculated value from ks."""
        try:
            self.chi_stat, self.chi_p = stats.chisquare(self.pred, self.true)
        except (ValueError, TypeError):
            self.chi_p = np.nan
            self.chi_stat = np.nan
        return self.chi_p

    @property
    def explained_var_score(self):
        """Return calculated explained_variance_score."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.explained_variance_score(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            return 1 - np.var(self.residuals) / np.var(self.true)

    @property
    def max_err(self):
        """Return calculated max_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.max_error(self.true, self.pred, sample_weight=self.weight)
        else:
            return abs(self.residuals)

    @property
    def mean_abs_err(self):
        """Return calculated mean_absolute_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_absolute_error(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            return np.average(self.max_err, weights=self.weight, axis=0)

    @property
    def median_abs_err(self):
        """Return calculated median_absolute_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.median_absolute_error(self.true, self.pred)
        else:
            return np.median(self.max_err)

    @property
    def mean_sq_err(self):
        """Return calculated mean_squared_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_squared_error(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            return np.average(self.sq_err, weights=self.weight, axis=0)

    @property
    def root_mean_sq_err(self):
        """Return calculated mean_squared_error."""
        return np.sqrt(self.mean_sq_err)

    @property
    def root_mean_abs_err(self):
        """Return calculated root mean_squared_error."""
        return np.sqrt(self.mean_abs_err)

    @property
    def mean_abs_perc_err(self):
        """Return calculated mean_absolute_percentage_error."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_absolute_percentage_error(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            array = self.max_err / abs(
                np.where(self.true != 0, self.true, np.finfo(np.float64).eps)
            )
            return np.average(array, weights=self.weight, axis=0)

    @property
    def mean_sq_log_err(self):
        """Return calculated mean_squared_log_error."""
        if (
            not np.iscomplexobj(self.true)
            and not np.iscomplexobj(self.pred)
            and (self.true > 0).all()
            and (self.pred > 0).all()
        ):
            return metrics.mean_squared_log_error(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            array = np.square(np.log(1 + self.true) - np.log(1 + self.pred))
            return np.average(array, weights=self.weight, axis=0)

    @property
    def mean_poisson_dev(self):
        """Return calculated mean_poisson_deviance."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_poisson_deviance(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            array = 2 * (
                self.true * np.log(self.true / self.pred) + self.pred - self.true
            )
            return np.average(array, weights=self.weight, axis=0)

    @property
    def mean_gamma_dev(self):
        """Return calculated mean_gamma_deviance."""
        if not np.iscomplexobj(self.true) and not np.iscomplexobj(self.pred):
            return metrics.mean_gamma_deviance(
                self.true, self.pred, sample_weight=self.weight
            )
        else:
            array = 2 * (np.log(self.pred / self.true) + self.true / self.pred - 1)
            return np.average(array, weights=self.weight, axis=0)

    def pcov_calc(self, params, func=None, eps=None, args=[], **kwargs):
        """Calculate parameter covariance."""
        eps = kwargs.get("eps", np.sqrt(np.finfo(float).eps))

        if func is None and hasattr(self, "func"):
            func = getattr(self, "func")
        elif func is callable:
            self.func = func

        self.n_param = len(params)
        if (
            isinstance(eps, (float, np.float, int, np.integer))
            or len(eps) == self.n_param
        ):
            self.jac = np.array(
                [
                    optimize.approx_fprime(params, func, eps, indep, *args)
                    for indep in self.indep
                ]
            )
        elif len(eps) == len(self.indep):
            self.jac = np.array(
                [
                    optimize.approx_fprime(params, func, eps[n], self.indep[n], *args)
                    for n in range(len(eps))
                ]
            )
        else:
            self.jac = np.array(
                [
                    optimize.approx_fprime(params, func, eps[0], indep, *args)
                    for indep in self.indep
                ]
            )
        _, s, vh = linalg.svd(self.jac)
        threshold = np.finfo(float).eps * max(self.jac.shape) * s[0]
        s = s[s > threshold]
        vh = vh[: s.size]

        self.pcov = np.dot(vh.T / s**2, vh)
        self.perr = np.sqrt(np.diag(self.pcov))

        return self.perr

    def bands(self, params, intervals=0.1):
        """Calculate. generic discription."""
        if isinstance(intervals, (float, np.float)):
            intervals = params * intervals
        if intervals.shape == params.shape:
            intervals = np.array([params + intervals, params - intervals])

        inters = [intervals[n, :].copy() for n in range(2)]
        for n in range(len(params)):
            high = intervals[0, :].copy()
            low = intervals[1, :].copy()
            for m in range(len(params)):
                high[n - m] = intervals[1, n - m]
                low[n - m] = intervals[0, n - m]
                if not np.array([n.all() for n in list(np.isin(inters, high))]).any():
                    inters.append(high.copy())
                if not np.array([n.all() for n in list(np.isin(inters, low))]).any():
                    inters.append(low.copy())
        return inters
