# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""


import numpy as np
import sympy as sp
from typing import Union, Optional, Tuple, List

from ..functions import (
    get_const,
    has_units,
    has_arrays,
    pick_math_module,
    all_symbols,
)

from .general import inv_sum_invs, erfc


# %% Electrical
def capacitance(epsilon_r: float, A: float, L: float, **kwargs) -> float:
    """
    Calculate capacitance.
    EQ: C = (ε_r * ε_0 * A) / L
    
    Args:
        epsilon_r (float): Relative permittivity.
        A (float): Area (cm²).
        L (float): Length (cm).
        **kwargs: Additional arguments.
    
    Returns:
        float: Capacitance (F).
    """
    arg_in = vars().copy()
    epsilon_0 = kwargs.get("vacuum_permittivity", None)
    if epsilon_0 is None:
        w_units = has_units(arg_in)
        symbolic = all_symbols(arg_in)
        epsilon_0 = get_const("vacuum_permittivity", *([True] if symbolic else [w_units, ["farad", "cm"]]))

    res = epsilon_r * epsilon_0 * A / L

    return res


def resistance(rho: float, A: float, L: float) -> float:
    """
    Calculate resistance.
    EQ: R = (ρ * L) / A
    
    Args:
        rho (float): Resistivity (ohm·cm).
        A (float): Area (cm²).
        L (float): Length (cm).
    
    Returns:
        float: Resistance (ohm).
    """
    res = rho * L / A
    return res


def ohms_law(V: float, R: float) -> float:
    """
    Calculate current from Ohm's Law.
    EQ: I = V / R
    
    Args:
        V (float): Voltage (V).
        R (float): Resistance (ohm).
    
    Returns:
        float: Current (A).
    """
    res = V / R
    return res


def voltage_divider(R: Union[dict, list, tuple, np.ndarray], V: float = 1, R0: float = 1) -> float:
    """
    Calculate the component voltage from the voltage divider.
    EQ: V_out = V * (R0 / (R_total))
    
    Args:
        R (dict, list, tuple, np.ndarray): Resistances.
        V (float, optional): Voltage (V). Default is 1.
        R0 (float, optional): Reference resistance. Default is 1.
    
    Returns:
        float: Component voltage (V).
    """
    if isinstance(R, dict):
        Rt = sum(R.values())
        R0 = R.get(R0, R0)
    elif isinstance(R, (list, tuple, np.ndarray)):
        Rt = sum(R)
        try:
            R0 = R[int(R0)]
        except (ValueError, IndexError):
            if R0 not in R:
                R0 = 1
    else:
        return None
    res = V * R0 / Rt
    return res


def sheet_resistivity(doping: float, thickness: float, dopant: Optional[str] = None) -> float:
    """
    Calculate the sheet resistivity from doping.
    EQ: ρ_sheet = 1 / (q * doping * mobility * thickness)
    
    Args:
        doping (float): Doping concentration (cm⁻³).
        thickness (float): Thickness (cm).
        dopant (str, optional): Dopant type.
    
    Returns:
        float: Sheet resistivity (ohm/sq).
    """
    arg_in = vars().copy()
    if dopant is not None:
        mob = mobility_masetti(doping, dopant)
    else:
        mob = 300  # assume constant mobility

    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))

    res = 1 / (q * doping * mob * thickness)

    return res


def conductivity(n: float, p: float, ue: float, uh: float) -> float:
    """
    Calculate the conductivity of a material.
    EQ: σ = q * (ue * n + uh * p)
    
    Args:
        n (float): Electron concentration (cm⁻³).
        p (float): Hole concentration (cm⁻³).
        ue (float): Electron mobility (cm²/Vs).
        uh (float): Hole mobility (cm²/Vs).
    
    Returns:
        float: Conductivity (S/cm).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))

    res = q * ue * n + q * uh * p

    return res


def resistivity_Si_n(Ndonor: float) -> float:
    """
    Calculate the resistivity of n-type silicon.
    EQ: ρ = 1 / (q * (μ_n * N_donor + μ_p * n_minority))
    
    Args:
        Ndonor (float): Donor concentration (cm⁻³).
    
    Returns:
        float: Resistivity (ohm·cm).
    """
    arg_in = vars().copy()
    n_minority = ni_Si() ** 2 / Ndonor

    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))

    res = 1 / (
        (q * mobility_thurber(Ndonor, False) * Ndonor)
        + (q * mobility_thurber(n_minority, False, False) * n_minority)
    )

    return res


def resistivity_Si_p(Nacceptor: float) -> float:
    """
    Calculate the resistivity of p-type silicon.
    EQ: ρ = 1 / (q * (μ_p * N_acceptor + μ_n * n_minority))
    
    Args:
        Nacceptor (float): Acceptor concentration (cm⁻³).
    
    Returns:
        float: Resistivity (ohm·cm).
    """
    arg_in = vars().copy()
    n_minority = ni_Si() ** 2 / Nacceptor

    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))

    res = 1 / (
        (q * mobility_thurber(Nacceptor) * Nacceptor)
        + (q * mobility_thurber(n_minority, True, False) * n_minority)
    )
    return res


def resistivity(N: float, dopant: str, W: float) -> float:
    """
    Calculate the resistivity of silicon.
    EQ: ρ = 1 / (q * μ * N * W)
    
    Args:
        N (float): Doping concentration (cm⁻³).
        dopant (str): Dopant type.
        W (float): Width (cm).
    
    Returns:
        float: Resistivity (ohm·cm).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    res = 1 / (q * mobility_generic(N, dopant) * N * W)
    return res


# %% Semiconductors
def v_thermal(T: float = 298.15) -> float:
    """
    Calculate thermal voltage.
    EQ: V_th = k_B * T
    
    Args:
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Thermal voltage (V).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = k_B * T
    return res


def depletion_region(Na: float, Nd: float, T: float = 298.15) -> Tuple[float, float]:
    """
    Calculate depletion region thickness.
    EQ: x_n = sqrt((2 * ε_0 * ε_r * V_bi * Nd) / (q * Na * (Na + Nd)))
        x_p = sqrt((2 * ε_0 * ε_r * V_bi * Na) / (q * Nd * (Na + Nd)))
    
    Args:
        Na (float): Acceptor concentration (cm⁻³).
        Nd (float): Donor concentration (cm⁻³).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        tuple: Thicknesses (cm).
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, depletion_region)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    epsilon_0 = get_const("vacuum_permittivity", *([True] if symbolic else [w_units, ["farad", "cm"]]))
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    Vbi = k_B * T * nsp.log(Na * Nd / ni_Si(T) ** 2)

    pre = 2 * 11.8 * epsilon_0 / q * Vbi * (1 / (Na + Nd))
    xp = nsp.sqrt(pre * Nd / Na)
    xn = nsp.sqrt(pre * Na / Nd)

    return xn, xp


def probability_fermi_dirac(E: float, Ef: float, T: float = 298.15) -> float:
    """
    Calculate Fermi-Dirac probability.
    EQ: f(E) = 1 / (exp((E - Ef) / (k_B * T)) + 1)
    
    Args:
        E (float): Energy (eV).
        Ef (float): Fermi energy (eV).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Probability.
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    res = 1 / (nsp.exp((E - Ef) / (k_B * T)) + 1.0)
    return res


def probability_maxwell_boltzmann(E: float, Ef: float, T: float = 298.15) -> float:
    """
    Calculate Maxwell-Boltzmann probability.
    EQ: f(E) = exp(-(E - Ef) / (k_B * T))
    
    Args:
        E (float): Energy (eV).
        Ef (float): Fermi energy (eV).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Probability.
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = 1 / (nsp.exp((E - Ef) / (k_B * T)))
    return res


def probability_bose_einstein(E: float, Ef: float, T: float = 298.15) -> float:
    """
    Calculate Bose-Einstein probability.
    EQ: f(E) = 1 / (exp((E - Ef) / (k_B * T)) - 1)
    
    Args:
        E (float): Energy (eV).
        Ef (float): Fermi energy (eV).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Probability.
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = 1 / (nsp.exp((E - Ef) / (k_B * T)) - 1.0)
    return res


def equilibrium_carrier(doping: float, **kwargs) -> Tuple[float, float]:
    """
    Calculate equilibrium carrier concentrations.
    EQ: carrier = doping / (ni^2)
    
    Args:
        doping (float): Doping concentration (cm⁻³).
        **kwargs: Additional arguments.
    
    Returns:
        tuple: Majority and minority carrier concentrations (cm⁻³).
    
    Note:
        Strictly speaking N and ni just have to be in the same units but (cm-3 is almost always used.
    """
    ni = kwargs.get("ni", ni_Si(kwargs.get("T", 298.15)))
    carrier = doping / (ni**2)
    return max(doping, carrier), min(doping, carrier)


def ni_Si(T: float = 298.15, narrowing: bool = True) -> float:
    """
    Calculate intrinsic carrier concentration of silicon.
    EQ: ni = 9.38e19 * (T / 300)^2 * exp(-6884 / T) (with narrowing)
        ni = 5.29e19 * (T / 300)^2.54 * exp(-6726 / T) (without narrowing)
    
    Args:
        T (float, optional): Temperature (K). Default is 298.15.
        narrowing (bool, optional): Band gap narrowing. Default is True.
    
    Returns:
        float: Intrinsic carrier concentration (cm⁻³).

    Notes:
        With narrowing calculates according to Sproul94 http://dx.doi.org/10.1063/1.357521
        Without narrowing calculates according to Misiakos93 http://dx.doi.org/10.1063/1.354551
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    if narrowing:
        res = 9.38e19 * (T / 300) * (T / 300) * nsp.exp(-6884 / T)
    else:
        res = 5.29e19 * (T / 300) ** 2.54 * nsp.exp(-6726 / T)
    return res


def ni_eff(N_D: float, N_A: float, delta_n: float, T: float = 298.15) -> float:
    """
    Calculate effective intrinsic carrier concentration.
    EQ: ni_eff = ni0 * exp((dEc + dEv) / (2 * k_B * T))
    
    Args:
        N_D (float): Donor concentration (cm⁻³).
        N_A (float): Acceptor concentration (cm⁻³).
        delta_n (float): Excess carrier density (cm⁻³).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Effective intrinsic carrier concentration (cm⁻³).
    
    Notes:
        calculation of the effective intrinsic concentration n_ieff including BGN
    according to Altermatt JAP 2003
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, ni_eff)
    if lam_res:
        return symb_var

    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    # n_i without BGN according to Misiakos93, parameterization fits very well
    # to value of Altermatt2003 at 300K
    ni0 = ni_Si(T, False)

    ni = ni0  # ni0 as starting value for n_ieff for calculation of n0 & p0

    n0 = np.where(N_D > N_A, N_D, N_A / ni**2)
    p0 = np.where(N_D > N_A, N_D / ni**2, N_A)

    # self-conistent iterative calculation of n_ieff

    for i in range(5):  # lazy programmer as it converges pretty fast anyway
        n = n0 + delta_n
        p = p0 + delta_n
        dEc, dEv = bandgap_schenk(n, p, N_A, N_D, delta_n, T)
        ni = ni0 * np.exp(
            (dEc + dEv) / (2 * (k_B * T))
        )  # there is something wrong here as the units don't match up.
        n0 = np.where(N_D > N_A, N_D, N_A / ni**2)
        p0 = np.where(N_D > N_A, N_D / ni**2, N_A)

    # print('iterations',ni)
    # if isinstance(ni, nsp.Number):
    #     return float(ni)
    return ni


def bandgap_paessler(T: float = 298.15) -> float:
    """
    Calculate bandgap of silicon.
    EQ: Eg0 = Eg0_T0 - α * θ * ((1 - 3 * δ^2) / (exp(θ / T) - 1) + 3 / 2 * δ^2 * (wurzel - 1))
    
    Args:
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Bandgap (eV).
    
    Notes:
        Calculation according to Paessler2001, Code adapted from Richter at Fraunhofer ISE https://doi.org/10.1103/PhysRevB.66.085201
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    lam_res, symb_var = has_arrays(arg_in, bandgap_paessler)
    if lam_res:
        return symb_var

    # constants from Table I on page 085201-7
    alpha_ = 3.23 * 0.0001  # (eV/K)
    deg_ = 446  # (K)
    delta_ = 0.51
    Eg0_T0 = 1.17  # eV     band gap of Si at 0 K

    Tdelta = 2 * T / deg_
    wurzel = (
        1
        + nsp.pi**2 / (3 * (1 + delta_**2)) * Tdelta**2
        + (3 * delta_**2 - 1) / 4 * Tdelta**3
        + 8 / 3 * Tdelta**4
        + Tdelta**6
    ) ** (1 / 6)
    Eg0 = Eg0_T0 - alpha_ * deg_ * (
        (1 - 3 * delta_**2) / (nsp.exp(deg_ / T) - 1) + 3 / 2 * delta_**2 * (wurzel - 1)
    )
    # if isinstance(Eg0, nsp.Number):
    #     return float(Eg0)
    return Eg0


def bandgap_schenk(n_e: float, n_h: float, N_D: float, N_A: float, delta_n: float, T: float = 298.15) -> Tuple[float, float]:
    """
    Calculate bandgap narrowing in silicon.
    EQ: dE_gap = -Ry_ex * (delta_xc + delta_i)
    
    Args:
        n_e (float): Electron density (cm⁻³).
        n_h (float): Hole density (cm⁻³).
        N_D (float): Donor concentration (cm⁻³).
        N_A (float): Acceptor concentration (cm⁻³).
        delta_n (float): Excess carrier density (cm⁻³).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        tuple: Bandgap narrowing (eV).

    Notes:
        Band-gap narrowing after Schenk 1998, JAP 84(3689))
        model descriped very well in K. McIntosh IEEE PVSC 2010
        model confirmed by Glunz2001 & Altermatt2003
        nomenclatur and formula no. according to McIntosh2010, table no. according to Schenk1998
        ==========================================================================
        Input parameters:

        ==========================================================================
        Code adapted from Richter at Fraunhofer ISE
        http://dx.doi.org/10.1063%2F1.368545
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, bandgap_schenk)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    # Silicon material parameters (table 1)
    g_e = 12  # degeneracy factor for electrons
    g_h = 4  # degeneracy factor for holes
    alfa_e = 0.5187  # µ*/m_e
    alfa_h = 0.4813  # µ*/m_h
    Ry_ex = 0.01655  # eV    excitonic Rydberg constant
    alfa_ex = 0.0000003719  # cm     excitonic Bohr radius

    # Parameters for Pade-Approximation (tab. 2 & 3)
    b_e = 8
    b_h = 1
    c_e = 1.3346
    c_h = 1.2365
    d_e = 0.893
    d_h = 1.153
    p_e = 7 / 30
    p_h = 7 / 30
    h_e = 3.91
    h_h = 4.2
    j_e = 2.8585
    j_h = 2.9307
    k_e = 0.012
    k_h = 0.19
    q_e = 3 / 4
    q_h = 1 / 4

    # ==========================================================================
    # pre-calculations:
    F = ((k_B * T)) / Ry_ex  # eq. 29
    a3 = alfa_ex**3

    # Normalizing of the densities
    n_e *= a3
    n_h *= a3
    N_D *= a3
    N_A *= a3

    # for eq. 33 (normalized)
    n_sum_xc = n_e + n_h
    n_p_xc = alfa_e * n_e + alfa_h * n_h

    # for eq. 37 (normalized)
    n_sum_i = N_D + N_A  # eq.39 bzw. eq. 29
    n_p_i = alfa_e * N_D + alfa_h * N_A  # eq.39 bzw. eq. 29

    Ui = n_sum_i**2 / F**2  # eq. 38
    n_ionic = n_sum_i  # McIntosh2010

    # exchange quasi-partical shift Eq33:
    delta_xc_h = -(
        (4 * nsp.pi) ** 3
        * n_sum_xc**2
        * (
            (48 * n_h / (nsp.pi * g_h)) ** (1 / 3)
            + c_h * nsp.log(1 + d_h * n_p_xc**p_h)
        )
        + (8 * nsp.pi * alfa_h / g_h) * n_h * F**2
        + nsp.sqrt(8 * nsp.pi * n_sum_xc) * F ** (5 / 2)
    ) / (
        (4 * nsp.pi) ** 3 * n_sum_xc**2
        + F**3
        + b_h * nsp.sqrt(n_sum_xc) * F**2
        + 40 * n_sum_xc ** (3 / 2) * F
    )
    delta_xc_e = -(
        (4 * nsp.pi) ** 3
        * n_sum_xc**2
        * (
            (48 * n_e / (nsp.pi * g_e)) ** (1 / 3)
            + c_e * nsp.log(1 + d_e * n_p_xc**p_e)
        )
        + (8 * nsp.pi * alfa_e / g_e) * n_e * F**2
        + nsp.sqrt(8 * nsp.pi * n_sum_xc) * F ** (5 / 2)
    ) / (
        (4 * nsp.pi) ** 3 * n_sum_xc**2
        + F**3
        + b_e * nsp.sqrt(n_sum_xc) * F**2
        + 40 * n_sum_xc ** (3 / 2) * F
    )

    # ionic quasi-partical shift Eq37:
    delta_i_h = (
        -n_ionic
        * (1 + Ui)
        / (
            nsp.sqrt(0.5 * F * n_sum_i / nsp.pi)
            * (1 + h_h * nsp.log(1 + nsp.sqrt(n_sum_i) / F))
            + j_h * Ui * n_p_i**0.75 * (1 + k_h * n_p_i**q_h)
        )
    )
    delta_i_e = (
        -n_ionic
        * (1 + Ui)
        / (
            nsp.sqrt(0.5 * F * n_sum_i / nsp.pi)
            * (1 + h_e * nsp.log(1 + nsp.sqrt(n_sum_i) / F))
            + j_e * Ui * n_p_i**0.75 * (1 + k_e * n_p_i**q_e)
        )
    )

    # rescale BGN
    dE_gap_h = -Ry_ex * (delta_xc_h + delta_i_h)
    dE_gap_e = -Ry_ex * (delta_xc_e + delta_i_e)
    # if isinstance(dE_gap_h, nsp.Number):
    #     dE_gap_h = float(dE_gap_h)
    # if isinstance(dE_gap_e, nsp.Number):
    #     dE_gap_e = float(dE_gap_e)
    return dE_gap_e, dE_gap_h


# %%  Mobilities
def mobility_generic(N: float, dopant: str) -> float:
    """
    Calculate carrier mobility in silicon.
    EQ: μ = umin + (umax - umin) / (1 + (N / Nref)^a)
    
    Args:
        N (float): Doping concentration (cm⁻³).
        dopant (str): Dopant type.
    
    Returns:
        float: Mobility (cm²/Vs).
        
    Notes:
        Return the mobility of carriers in silicon according to
        the model of Thurbur as a function of doping
        Where:
        N - doping level (cm-3)
        Data is included for specific dopant values as given in mini-project 3
        https://archive.org/details/relationshipbetw4006thur
    
    """ 
    # if "A" in dopant:
    umin = 52.2
    umax = 1417
    Nref = 9.68e16
    a = 0.68
    if "P" in dopant:
        umin = 68.5
        umax = 1414
        Nref = 9.20e16
        a = 0.711
    if "B" in dopant:
        umin = 44.9
        umax = 470.5
        Nref = 2.23e17
        a = 0.719
    return umin + (umax - umin) / (1 + ((N / Nref) ** a))


def mobility_thurber(N: float, p_type: bool = True, majority: bool = True) -> float:
    """
    Calculate carrier mobility using Thurber model.
    EQ: μ = umin + (umax - umin) / (1 + (N / Nref)^a)
    
    Args:
        N (float): Doping concentration (cm⁻³).
        p_type (bool, optional): p-type material. Default is True.
        majority (bool, optional): Majority carriers. Default is True.
    
    Returns:
        float: Mobility (cm²/Vs).

    Notes:
        Return the mobility of carriers in silicon according to the model of Thurbur
        as a function of doping
        https://archive.org/details/relationshipbetw4006thur
    """
    i = 2 * p_type + majority
    # n-type minority, n-type majority, p-type minority, p-type majority
    umax = [1417, 1417, 470, 470][i]
    umin = [160, 60, 155, 37.4][i]
    Nref = [5.6e16, 9.64e16, 1e17, 2.82e17][i]
    a = [0.647, 0.664, 0.9, 0.642][i]
    return umin + (umax - umin) / (1 + ((N / Nref) ** a))


def mobility_masetti(N: float, dopant: int = 0) -> float:
    """
    Calculate carrier mobility using Masetti model.
    EQ: μ = μmin + (μmax - μmin) / (1 + (N / Nref1)^a) - u1 / (1 + (Nref2 / N)^b)
    
    Args:
        N (float): Doping concentration (cm⁻³).
        dopant (int, optional): Dopant type. Default is 0.
    
    Returns:
        float: Mobility (cm²/Vs).
        
    Notes:
        mobility model from Masetti DOI: 10.1109/T-ED.1983.21207
        
    """
    # if dopant == 0:
    µmax = 1414
    µmin = 68.5
    u1 = 56.1
    Nref1 = 9.20e16
    Nref2 = 3.41e20
    a = 0.711
    b = 1.98
    if dopant == 1:
        µmax = 470.5
        µmin = 44.9
        u1 = 29.0
        Nref1 = 2.23e17
        Nref2 = 6.1e20
        a = 0.719
        b = 1.98
    return (
        µmin + (µmax - µmin) / (1 + ((N / Nref1) ** a)) - u1 / (1 + ((Nref2 / N) ** b))
    )


def mobility_klassen(Nd: float, Na: float, delta_n: float = 1, T: float = 298.16) -> Tuple[float, float]:
    """
    Calculate carrier mobility using Klassen model.
    EQ: μe = 1 / (1 / μ_eL + 1 / μe_Dah)
        μh = 1 / (1 / μ_hL + 1 / μh_Dae)
    
    Args:
        Nd (float): Donor concentration (cm⁻³).
        Na (float): Acceptor concentration (cm⁻³).
        delta_n (float, optional): Excess carrier density (cm⁻³). Default is 1.
        T (float, optional): Temperature (K). Default is 298.16.
    
    Returns:
        tuple: Electron and hole mobilities (cm²/Vs).
    """
    s1 = 0.89233
    s2 = 0.41372
    s3 = 0.19778
    s4 = 0.28227
    s5 = 0.005978
    s6 = 1.80618
    s7 = 0.72169
    r1 = 0.7643
    r2 = 2.2999
    r3 = 6.5502
    r4 = 2.367
    r5 = -0.01552
    r6 = 0.6478
    fCW = 2.459
    fBH = 3.828
    mh_me = 1.258
    me_m0 = 1

    T = 298.16
    n0, p0 = equilibrium_carrier(Nd)

    cA = 0.5
    cD = 0.21
    Nref_A = 7.20e20
    Nref_D = 4.00e20

    p = p0 + delta_n
    n = n0 + delta_n
    cc = p + n

    Za_Na = 1 + 1 / (cA + (Nref_A / Na) ** 2)
    Zd_Nd = 1 + 1 / (cD + (Nref_D / Nd) ** 2)

    Na_h = Za_Na * Na
    Nd_h = Zd_Nd * Nd

    boron_µmax = 470.5
    boron_µmin = 44.9
    boron_Nref_1 = 2.23e17
    boron_alpha_ = 0.719
    boron_deg_ = 2.247

    phosphorus_µmax = 1414
    phosphorus_µmin = 68.5
    phosphorus_Nref_1 = 9.20e16
    phosphorus_alpha_ = 0.711
    phosphorus_deg_ = 2.285

    µ_eN = (
        phosphorus_µmax**2
        / (phosphorus_µmax - phosphorus_µmin)
        * (T / 300) ** (3 * phosphorus_alpha_ - 1.5)
    )
    µ_hN = (
        boron_µmax**2 / (boron_µmax - boron_µmin) * (T / 300) ** (3 * boron_alpha_ - 1.5)
    )

    µ_ec = (
        phosphorus_µmax
        * phosphorus_µmin
        / (phosphorus_µmax - phosphorus_µmin)
        * (300 / T) ** 0.5
    )
    µ_hc = boron_µmax * boron_µmin / (boron_µmax - boron_µmin) * (300 / T) ** 0.5

    Ne_sc = Na_h + Nd_h + p
    Nh_sc = Na_h + Nd_h + n

    PBHe = 1.36e20 / cc * me_m0 * (T / 300) ** 2
    PBHh = 1.36e20 / cc * mh_me * (T / 300) ** 2

    PCWe = 3.97e13 * (1 / (Zd_Nd**3 * (Nd_h + Na_h + p)) * ((T / 300) ** 3)) ** (
        2 / 3
    )
    PCWh = 3.97e13 * (1 / (Za_Na**3 * (Nd_h + Na_h + n)) * ((T / 300) ** 3)) ** (
        2 / 3
    )

    Pe = 1 / (fCW / PCWe + fBH / PBHe)
    Ph = 1 / (fCW / PCWh + fBH / PBHh)

    G_Pe = (
        1
        - s1 / ((s2 + (1 / me_m0 * 300 / T) ** s4 * Pe) ** s3)
        + s5 / (((me_m0 * 300 / T) ** s7 * Pe) ** s6)
    )
    G_Ph = (
        1
        - s1 / ((s2 + (1 / (me_m0 * mh_me) * T / 300) ** s4 * Ph) ** s3)
        + s5 / (((me_m0 * mh_me * 300 / T) ** s7 * Ph) ** s6)
    )

    F_Pe = (r1 * Pe**r6 + r2 + r3 / mh_me) / (Pe**r6 + r4 + r5 / mh_me)
    F_Ph = (r1 * Ph**r6 + r2 + r3 * mh_me) / (Ph**r6 + r4 + r5 * mh_me)

    Ne_sc_eff = Nd_h + G_Pe * Na_h + p / F_Pe
    Nh_sc_eff = Na_h + G_Ph * Nd_h + n / F_Ph

    # Lattice Scattering
    µ_eL = phosphorus_µmax * (300 / T) ** phosphorus_deg_
    µ_hL = boron_µmax * (300 / T) ** boron_deg_

    µe_Dah = µ_eN * Ne_sc / Ne_sc_eff * (
        phosphorus_Nref_1 / Ne_sc
    ) ** phosphorus_alpha_ + µ_ec * ((p + n) / Ne_sc_eff)
    µh_Dae = µ_hN * Nh_sc / Nh_sc_eff * (boron_Nref_1 / Nh_sc) ** boron_alpha_ + µ_hc * (
        (p + n) / Nh_sc_eff
    )

    µe = 1 / (1 / µ_eL + 1 / µe_Dah)
    µh = 1 / (1 / µ_hL + 1 / µh_Dae)

    return µe, µh


def mobility_einstein(D: float, z: int = 1, T: float = 298.15) -> float:
    """
    Calculate mobility or diffusivity using Einstein relation.
    EQ: μ = D * z / (k_B * T)
    
    Args:
        D (float): Diffusivity (cm²/s).
        z (int, optional): Charge number. Default is 1.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Mobility (cm²/Vs).
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, mobility_einstein)
    if lam_res:
        return symb_var

    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = D * z / (k_B * T)

    return res

# %% Diffusion



def diffusion_length(t: float, D: float) -> float:
    """
    Calculate carrier diffusion length.
    EQ: L = sqrt(t * D)
    
    Args:
        t (float): Time (s).
        D (float): Diffusivity (cm²/s).
    
    Returns:
        float: Diffusion length (cm).
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, diffusion_length)
    if lam_res:
        return symb_var
    nsp = pick_math_module(arg_in)
    res = nsp.sqrt(t * D)
    return res

def debye_length(C: Union[float, List[float]], z: Union[int, List[int]], epsilon_r: float, T: float = 298.15) -> float:
    """
    Calculate Debye length.
    EQ: λ_D = sqrt(ε_r * ε_0 * k_B * T / (q^2 * C * z^2))
    
    Args:
        C (float): Concentration (cm⁻³).
        z (int): Charge number.
        epsilon_r (float): Relative permittivity.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Debye length (cm).
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, debye_length)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    epsilon_0 = get_const("vacuum_permittivity", *([True] if symbolic else [w_units, ["farad", "cm"]]))
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["joule", "K"]]))

    if isinstance(C, (tuple, list)):
        if not isinstance(z, (tuple, list)):
            z = [z] * len(C)
        charges = sum([C[n] * (q * z[n]) ** 2 for n in range(len(C))])
    else:
        charges = C * (q * z) ** 2
    res = nsp.sqrt(epsilon_r * epsilon_0 * k_B * T / (charges))

    return res


def bjerrum_length(epsilon_r: float, T: float = 298.15) -> float:
    """
    Calculate Bjerrum length.
    EQ: λ_B = q^2 / (ε_r * ε_0 * k_B * T)
    
    Args:
        epsilon_r (float): Relative permittivity.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Bjerrum length (cm).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    epsilon_0 = get_const("vacuum_permittivity", *([True] if symbolic else [w_units, ["farad", "cm"]]))
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["joule", "K"]]))

    res = q**2 / (epsilon_r * epsilon_0 * k_B * T)

    return res

def characteristic_length(E: float, D: float, t: float, z: int = 1, T: float = 298.15) -> float:
    """
    Calculate characteristic length.
    EQ: L_c = 2 * sqrt(D * t) + μ * E * t
    
    Args:
        E (float): Electric field (V/cm).
        D (float): Diffusivity (cm²/s).
        t (float): Time (s).
        z (int, optional): Charge number. Default is 1.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Characteristic length (cm).
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, characteristic_length)
    if lam_res:
        return symb_var

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    mob = D * z / (k_B * T)

    res = 2 * nsp.sqrt(D * t) + mob * E * t

    return res

def nernst_planck_fundamental_sol(C_0: float, x: float, D: float, t: float) -> float:
    """
    Calculate Nernst-Planck fundamental solution.
    EQ: C(x, t) = C_0 * erfc(x / (2 * sqrt(D * t)))
    
    Args:
        C_0 (float): Initial concentration (cm⁻³).
        x (float): Position (cm).
        D (float): Diffusivity (cm²/s).
        t (float): Time (s).
    
    Returns:
        float: Concentration (cm⁻³).
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, nernst_planck_fundamental_sol)
    if lam_res:
        return symb_var
    nsp = pick_math_module(arg_in)

    res = C_0 * erfc((x) / (2 * nsp.sqrt(D * t)))
    return res

def nernst_planck_analytic_sol(C_0: float, x: float, L: float, E: float, D: float, t: float, z: int = 1, T: float = 298.15) -> float:
    """
    Calculate Nernst-Planck analytic solution.
    EQ: C(x, t) = (C_0 / (2 * erfc(-μ * E * t / (2 * sqrt(D * t)))) * (erfc((x - μ * E * t) / (2 * sqrt(D * t))) + erfc(-(x - 2 * L + μ * E * t) / (2 * sqrt(D * t))))
    
    Args:
        C_0 (float): Initial concentration (cm⁻³).
        x (float): Position (cm).
        L (float): Length (cm).
        E (float): Electric field (V/cm).
        D (float): Diffusivity (cm²/s).
        t (float): Time (s).
        z (int, optional): Charge number. Default is 1.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Concentration (cm⁻³).
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, nernst_planck_analytic_sol)
    if lam_res:
        return symb_var

    if x is None:
        x = L
    elif L is None:
        L = x

    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    mob = D * z / (k_B * T)

    term_A1 = erfc((x - mob * E * t) / (2 * nsp.sqrt(D * t)))
    term_A2 = erfc(-(x - 2 * L + mob * E * t) / (2 * nsp.sqrt(D * t)))
    term_B = erfc(-mob * E * t / (2 * nsp.sqrt(D * t)))
    res = (C_0 / (2 * term_B)) * (term_A1 + term_A2)

    return res

def screened_permitivity(epsilon_r: float, kappa: float, x: float = 1) -> float:
    """
    Calculate screened permittivity.
    
    Args:
        epsilon_r (float): Relative permittivity.
        kappa (float): Screening parameter.
        x (float, optional): Distance (cm). Default is 1.
    
    Returns:
        float: Screened permittivity (F/cm).
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    epsilon_0 = get_const("vacuum_permittivity", *([True] if symbolic else [w_units, ["farad", "cm"]]))

    res = epsilon_r * epsilon_0 * nsp.exp(kappa * x)

    return res

def poisson_rhs(C: Union[float, List[float]], z: int, epsilon_r: float) -> Union[float, List[float]]:
    """
    Calculate Poisson right-hand side.
    
    Args:
        C (float): Concentration (cm⁻³).
        z (int): Charge number.
        epsilon_r (float): Relative permittivity.
    
    Returns:
        float: Poisson RHS (V/cm²).
    """
    if isinstance(C, (tuple,list)):
        return [poisson_rhs(var, z, epsilon_r) for var in C]

    arg_in = vars().copy()

    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    epsilon_0 = get_const("vacuum_permittivity", *([True] if symbolic else [w_units, ["farad", "cm"]]))

    res = q * z * C / (epsilon_r * epsilon_0)
    return res


# %% Recombination & Lifetime
def U_radiative(n: float, p: float) -> float:
    """
    Calculate radiative recombination rate.
    
    Args:
        n (float): Electron concentration (cm⁻³).
        p (float): Hole concentration (cm⁻³).
    
    Returns:
        float: Radiative recombination rate (cm⁻³/s).
    """
    B_rad = 4.73e-15
    U_radiative = n * p * B_rad
    return U_radiative


def U_radiative_alt(n0: float, p0: float, delta_n: float, T: float = 298.15) -> float:
    """
    Calculate alternative radiative recombination rate.
    
    Args:
        n0 (float): Initial electron concentration (cm⁻³).
        p0 (float): Initial hole concentration (cm⁻³).
        delta_n (float): Excess carrier density (cm⁻³).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Radiative recombination rate (cm⁻³/s).
    """
    n_p = n0 + p0 + 2 * delta_n
    n = n0 + delta_n
    p = p0 + delta_n
    B_low = 4.73e-15
    b_min = 0.2 + (0 - 0.2) / (1 + (T / 320) ** 2.5)
    b1 = 1.5e18 + (10000000 - 1.5e18) / (1 + (T / 550) ** 3)
    b3 = 4e18 + (1000000000 - 4e18) / (1 + (T / 365) ** 3.54)
    B_rel = b_min + (1 - b_min) / (
        1 + (0.5 * n_p / b1) ** 0.54 + (0.5 * n_p / b3) ** 1.25
    )
    B_rad = B_low * B_rel
    U_radiative_alt = n * p * B_rad
    return U_radiative_alt


def U_SRH(n: float, p: float, Et: float, tau_n: float, tau_p: float, ni_eff: float = 8.5e9, T: float = 298.15) -> float:
    """
    Calculate Shockley-Read-Hall recombination rate.
    
    Args:
        n (float): Electron concentration (cm⁻³).
        p (float): Hole concentration (cm⁻³).
        Et (float): Trap energy level (eV).
        tau_n (float): Electron lifetime (s).
        tau_p (float): Hole lifetime (s).
        ni_eff (float, optional): Effective intrinsic carrier concentration (cm⁻³). Default is 8.5e9.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: SRH recombination rate (cm⁻³/s).
    """
    arg_in = vars().copy()
    nsp = pick_math_module(arg_in)
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    n1 = ni_eff * nsp.exp(Et / (k_B * T))
    p1 = ni_eff * nsp.exp(-Et / (k_B * T))
    res = (n * p - ni_eff**2) / (tau_p * (n + n1) + tau_n * (p + p1))
    return res


def U_auger_richter(n0: float, p0: float, delta_n: float, ni_eff: float) -> float:
    """
    Calculate Auger recombination rate.
    
    Args:
        n0 (float): Initial electron concentration (cm⁻³).
        p0 (float): Initial hole concentration (cm⁻³).
        delta_n (float): Excess carrier density (cm⁻³).
        ni_eff (float): Effective intrinsic carrier concentration (cm⁻³).
    
    Returns:
        float: Auger recombination rate (cm⁻³/s).

    Notes:
        https://doi.org/10.1016/j.egypro.2012.07.034
    """
    arg_in = vars().copy()
    lam_res, symb_var = has_arrays(arg_in, U_auger_richter)
    if lam_res:
        return symb_var
    nsp = pick_math_module(arg_in)

    B_n0 = 2.5e-31
    C_n0 = 13
    D_n0 = 3.3e17
    exp_n0 = 0.66
    B_p0 = 8.5e-32
    C_p0 = 7.5
    D_p0 = 7e17
    exp_p0 = 0.63
    C_dn = 3e-29
    D_dn = 0.92
    g_eeh = 1 + C_n0 * (1 - nsp.tanh((n0 / D_n0) ** exp_n0))
    g_ehh = 1 + C_p0 * (1 - nsp.tanh((p0 / D_p0) ** exp_p0))
    np_ni2 = (n0 + delta_n) * (p0 + delta_n) - ni_eff**2
    res = np_ni2 * (B_n0 * n0 * g_eeh + B_p0 * p0 * g_ehh + C_dn * delta_n**D_dn)
    return res


def U_low_doping(n0: float, p0: float, delta_n: float) -> float:
    """
    alculate recombination rate at low doping.
    
    Args:
        n0 (float): Initial electron concentration (cm⁻³).
        p0 (float): Initial hole concentration (cm⁻³).
        delta_n (float): Excess carrier density (cm⁻³).
    
    Returns:
        float: Recombination rate (cm⁻³/s).

    Notes:
        equation 21 in DOI: 10.1103/PhysRevB.86.165202
    """
    B_low = 4.73e-15
    n = n0 + delta_n
    p = p0 + delta_n
    U = delta_n / (
        n
        * p
        * (8.7e-29 * n0**0.91 + 6.0e-30 * p0**0.94 + 3.0e-29 * delta_n**0.92 + B_low)
    )
    return U


def U_surface(n: float, p: float, Sn: float, Sp: float, n1: float = 8.3e9, p1: float = 8.3e9, **kwargs) -> float:
    """
    Calculate surface recombination rate.
    
    Args:
        n (float): Electron concentration (cm⁻³).
        p (float): Hole concentration (cm⁻³).
        Sn (float): Surface recombination velocity for electrons (cm/s).
        Sp (float): Surface recombination velocity for holes (cm/s).
        n1 (float, optional): Electron concentration at surface (cm⁻³). Default is 8.3e9.
        p1 (float, optional): Hole concentration at surface (cm⁻³). Default is 8.3e9.
        **kwargs: Additional arguments.
    
    Returns:
        float: Surface recombination rate (cm⁻³/s).
    """
    ni = kwargs.get("ni", ni_Si(kwargs.get("T", 298.15)))
    U_surface = Sn * Sp * (n * p - ni**2) / (Sn * (n + n1) + Sp * (p + p1))
    return U_surface


def lifetime(U: float, delta_n: float) -> float:
    """
    Calculate carrier lifetime.
    
    Args:
        U (float): Recombination rate (cm⁻³/s).
        delta_n (float): Excess carrier density (cm⁻³).
    
    Returns:
        float: Carrier lifetime (s).
    """
    return delta_n / U


def lifetime_eff(*lifetimes: float) -> float:
    """
    Calculate effective carrier lifetime.
    
    Args:
        *lifetimes: Individual lifetimes (s).
    
    Returns:
        float: Effective carrier lifetime (s).
    """
    return inv_sum_invs(*lifetimes)


def lifetime_bulk(tau_eff: float, S: float, thickness: float) -> float:
    """
    Calculate bulk carrier lifetime.
    
    Args:
        tau_eff (float): Effective lifetime (s).
        S (float): Surface recombination velocity (cm/s).
        thickness (float): Thickness (cm).
    
    Returns:
        float: Bulk carrier lifetime (s).
    """
    return tau_eff - thickness / (2 * S)


def lifetime_minority(N: float, tao_0: float = 0.001, N_ref: float = 1e17) -> float:
    """
    Calculate minority carrier lifetime.
    
    Args:
        N (float): Doping concentration (cm⁻³).
        tao_0 (float, optional): Initial lifetime (s). Default is 0.001.
        N_ref (float, optional): Reference doping concentration (cm⁻³). Default is 1e17.
    
    Returns:
        float: Minority carrier lifetime (s).
    """
    return tao_0 / (1 + N / N_ref)


# not sure if I should keep these
def lifetime_auger(delta_n: float, Ca: float = 1.66e-30) -> float:
    """
    Calculate Auger lifetime.
    
    Args:
        delta_n (float): Excess carrier density (cm⁻³).
        Ca (float, optional): Auger coefficient. Default is 1.66e-30.
    
    Returns:
        float: Auger lifetime (s).
    """
    return 1 / (Ca * delta_n**2)


def lifetime_SRH(N: float, Nt: float, Et: float, sigma__n: float, sigma__p: float, delta_n: float, T: float = 298.15) -> None:
    """
    Calculate Shockley-Read-Hall lifetime.
    
    Args:
        N (float): Doping concentration (cm⁻³).
        Nt (float): Trap concentration (cm⁻³).
        Et (float): Trap energy level (eV).
        sigma__n (float): Electron capture cross-section (cm²).
        sigma__p (float): Hole capture cross-section (cm²).
        delta_n (float): Excess carrier density (cm⁻³).
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: SRH lifetime (s).
    """
    # TODO needs correction
    # p0 = N
    # n0 = (ni_Si(T) ** 2) / N
    # tau_n0 = 1 / (Nt * sigma__n * vth)
    # tau_p0 = 1 / (Nt * sigma__p * vth)
    # n1 = Nc * np.exp(-Et / Vt())
    # p1 = Nv * np.exp((-Et - Eg) / Vt())
    # k_ratio = sigma__n / sigma__p
    # tau_SRH = (tau_p0 * (n0 + n1 + delta_n) + tau_n0 * (p0 + p1 + delta_n)) / (n0 + p0 + delta_n)
    # return tau_SRH
    return print("non-functional")
