# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

Functions: 
- Quantum Efficiency: IQE_emitter, IQE_base, IQE_IBC_emitter1, IQE_IBC_emitter2, IQE_bulk, IQE_bulk2, IQE_depletion, IQE
- Current: J0_layer, J0_factor, J0, current2gen, I_diode, I_cell, I_cell_DD, I_cell_Rseries, I_cell_Rshunt
- Voltage: impliedV, V_Rseries, Voc, V_cell
- Cell Resistances: emitter_resistance, base_resistance, finger_resistance, finger_resistivity, finger_sheet, busbar_resistance, IBC_metal_resistance
- Cell Evaluations: cell_params, efficiency, finger_shading, finger_total_loss, FF, FF_ideal, normalised_Voc, FF_Rs, FF_Rsh, FF_RsRsh
- Silicon Material Properties: optical_properties, phos_active, phos_solubility

Sources/References:
- 
"""

from typing import Tuple

import numpy as np
import sympy as sp

from .physics import ni_Si, mobility_generic
from ..functions import get_const, has_units, all_symbols


# %% Quantum Efficiency
def IQE_emitter(ab: float, We: float, Le: float, De: float, Se: float, z: int = 1) -> float:
    """
    Calculate internal quantum efficiency of a solar cell emitter.
    EQ: IQE = (Le * ab / (ab^2 * Le^2 - 1)) * GF
    
    Args:
        ab (float): Absorption coefficient (/cm).
        We (float): Thickness of the emitter (cm).
        Le (float): Diffusion length of carriers in the emitter (cm).
        De (float): Diffusivity of carriers in the emitter (cm²/s).
        Se (float): Recombination at the front surface (cm/s).
        z (int, optional): Position. Default is 1.
    
    Returns:
        float: Internal quantum efficiency.
    
    Notes:
        Hovel, I think.
    """
    GF = (
        (Se * Le / De)
        + ab * Le
        - (sp.exp(-ab * We * z))
        * ((Se * Le / De) * sp.cosh(We / Le) + sp.sinh(We / Le))
    ) / ((Se * Le / De) * sp.sinh(We / Le) + sp.cosh(We / Le)) - Le * ab * sp.exp(
        -ab * We * z
    )
    res = (Le * ab / (ab * ab * Le * Le - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_base(ab: float, We_Wd: float, Wb: float, Lb: float, Db: float, Sb: float, z: int = 1) -> float:
    """
    Calculate quantum efficiency of the base of a solar cell.
    EQ: IQE = (exp(-ab * We_Wd * z)) * (Lb * ab / (ab^2 * Lb^2 - 1)) * GF
    
    Args:
        ab (float): Absorption coefficient (/cm).
        We_Wd (float): Junction depth (cm).
        Wb (float): Base width (cm).
        Lb (float): Diffusion length of minority carriers in the base (cm).
        Db (float): Diffusivity of minority carriers in the base (cm²/s).
        Sb (float): Surface recombination velocity (cm/s).
        z (int, optional): Position. Default is 1.
    
    Returns:
        float: Quantum efficiency.
    """
    GF = ab * Lb - (
        (Sb * Lb / Db) * (sp.cosh(Wb / Lb) - sp.exp(-ab * Wb * z))
        + sp.sinh(Wb / Lb)
        + Lb * ab * sp.exp(-ab * Wb * z)
    ) / ((Sb * Lb / Db) * sp.sinh(Wb / Lb) + sp.cosh(Wb / Lb))
    res = (sp.exp(-ab * We_Wd * z)) * (Lb * ab / (ab**2 * Lb**2 - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res

def IQE_IBC_emitter1(ab: float, We_Wd: float, We: float, Le: float, De: float, Se: float, z: int = 1) -> float:
    """
    Calculate internal quantum efficiency of an IBC solar cell emitter.
    EQ: IQE = (exp(-ab * We_Wd * z)) * (Le * ab / (ab^2 * Le^2 - 1)) * GF
    
    Args:
        ab (float): Absorption coefficient (/cm).
        We_Wd (float): Junction depth (cm).
        We (float): Thickness of the emitter (cm).
        Le (float): Diffusion length of carriers in the emitter (cm).
        De (float): Diffusivity of carriers in the emitter (cm²/s).
        Se (float): Recombination at the front surface (cm/s).
        z (int, optional): Position. Default is 1.
    
    Returns:
        float: Internal quantum efficiency.
    
    Notes:
        Hovel, I think.
    """
    GF = (
        (Se * Le / De)
        + ab * Le
        - abs(sp.exp(-ab * We * z))
        * ((Se * Le / De) * sp.cosh(We / Le) + sp.sinh(We / Le))
    ) / ((Se * Le / De) * sp.sinh(We / Le) + sp.cosh(We / Le)) - Le * ab * abs(
        sp.exp(-ab * We * z)
    )
    res = abs(sp.exp(-ab * We_Wd * z)) * (Le * ab / (ab * ab * Le * Le - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_IBC_emitter2(ab: float, We_Wd: float, Wb: float, Lb: float, Db: float, Sb: float, z: int = 1) -> float:
    """
    Calculate internal quantum efficiency of an IBC solar cell emitter.
    EQ: IQE = (exp(-ab * We_Wd * z)) * (Lb * ab / (ab^2 * Lb^2 - 1)) * GF
    
    Args:
        ab (float): Absorption coefficient (/cm).
        We_Wd (float): Junction depth (cm).
        Wb (float): Base width (cm).
        Lb (float): Diffusion length of minority carriers in the base (cm).
        Db (float): Diffusivity of minority carriers in the base (cm²/s).
        Sb (float): Surface recombination velocity (cm/s).
        z (int, optional): Position. Default is 1.
    
    Returns:
        float: Internal quantum efficiency.
    
    Notes:
        Hovel, I think.
    """
    GF = ab * Lb - (
        (Sb * Lb / Db) * (sp.cosh(Wb / Lb) - sp.exp(-ab * Wb * z))
        + sp.sinh(Wb / Lb)
        + Lb * ab * sp.exp(-ab * Wb * z)
    ) / ((Sb * Lb / Db) * sp.sinh(Wb / Lb) + sp.cosh(Wb / Lb))
    res = (sp.exp(-ab * We_Wd * z)) * (Lb * ab / (ab**2 * Lb**2 - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_bulk(ab: float, We_Wd: float, Wb: float, Lb: float, Db: float, Sb: float, z: int = 1) -> float:
    """
    Calculate quantum efficiency of the bulk of a solar cell.
    EQ: IQE = (exp(-ab * We_Wd * z)) * (Lb * ab / (ab^2 * Lb^2 - 1)) * GF
    
    Args:
        ab (float): Absorption coefficient (/cm).
        We_Wd (float): Junction depth (cm).
        Wb (float): Base width (cm).
        Lb (float): Diffusion length of minority carriers in the base (cm).
        Db (float): Diffusivity of minority carriers in the base (cm²/s).
        Sb (float): Surface recombination velocity (cm/s).
        z (int, optional): Position. Default is 1.
    
    Returns:
        float: Quantum efficiency.
    """
    GF = ab * Lb - (
        (Sb * Lb / Db) * (sp.cosh(Wb / Lb) - abs(sp.exp(-ab * Wb * z)))
        + sp.sinh(Wb / Lb)
        + Lb * ab * abs(sp.exp(-ab * Wb * z))
    ) / ((Sb * Lb / Db) * sp.sinh(Wb / Lb) + sp.cosh(Wb / Lb))
    res = abs(sp.exp(-ab * We_Wd * z)) * (Lb * ab / (ab**2 * Lb**2 - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_bulk2(ab: float, We_Wd: float, We: float, Le: float, De: float, Se: float, z: int = 1) -> float:
    """
    Calculate internal quantum efficiency of a solar cell emitter.
    EQ: IQE = (exp(-ab * We_Wd * z)) * (Le * ab / (ab^2 * Le^2 - 1)) * GF
    
    Args:
        ab (float): Absorption coefficient (/cm).
        We_Wd (float): Junction depth (cm).
        We (float): Thickness of the emitter (cm).
        Le (float): Diffusion length of carriers in the emitter (cm).
        De (float): Diffusivity of carriers in the emitter (cm²/s).
        Se (float): Recombination at the front surface (cm/s).
        z (int, optional): Position. Default is 1.
    
    Returns:
        float: Internal quantum efficiency.
    
    Notes:
        Hovel, I think.
    """
    GF = (
        (Se * Le / De)
        + ab * Le
        - sp.exp(-ab * We * z) * ((Se * Le / De) * sp.cosh(We / Le) + sp.sinh(We / Le))
    ) / ((Se * Le / De) * sp.sinh(We / Le) + sp.cosh(We / Le)) - Le * ab * sp.exp(
        -ab * We * z
    )
    res = (sp.exp(-ab * We_Wd * z)) * (Le * ab / (ab * ab * Le * Le - 1)) * GF
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE_depletion(ab: float, We: float, Wd: float) -> float:
    """
    Calculate internal quantum efficiency of the depletion region.
    EQ: IQE = exp(-ab * We) * (1 - exp(-ab * Wd))
    
    Args:
        ab (float): Absorption coefficient (/cm).
        We (float): Thickness of the emitter (cm).
        Wd (float): Depletion width (cm).
    
    Returns:
        float: Internal quantum efficiency.
    """
    res = sp.exp(-ab * We) * (1 - sp.exp(-ab * Wd))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def IQE(ab: float, Wd: float, Se: float, Le: float, De: float, We: float, Sb: float, Wb: float, Lb: float, Db: float) -> Tuple[float, float, float, float]:
    """
    Calculate total internal quantum efficiency of a solar cell.
    EQ: IQE_total = QEE + QEB + QED
    
    Args:
        ab (float): Absorption coefficient (/cm).
        Wd (float): Depletion width (cm).
        Se (float): Recombination at the front surface (cm/s).
        Le (float): Diffusion length of carriers in the emitter (cm).
        De (float): Diffusivity of carriers in the emitter (cm²/s).
        We (float): Thickness of the emitter (cm).
        Sb (float): Surface recombination velocity (cm/s).
        Wb (float): Base width (cm).
        Lb (float): Diffusion length of minority carriers in the base (cm).
        Db (float): Diffusivity of minority carriers in the base (cm²/s).
    
    Returns:
        tuple: Internal quantum efficiencies (emitter, base, depletion, total).
    """
    QEE = IQE_emitter(ab, We, Le, De, Se)
    QEB = IQE_base(ab, We + Wd, Wb, Lb, Db, Sb)
    QED = IQE_depletion(ab, We, Wd)
    IQEt = QEE + QEB + QED
    return QEE, QEB, QED, IQEt


# def QE2SR(wavelength, QE, R=0):
#     """'converts a QE in units to spectral response
#     given the wavelength (nm)"""
#     spectral_response = QE * wavelength * (1 - R) / 1239.8
#     return spectral_response


# def SR2QE(wavelength, spectral_response):
#     """convert SR (A/W) to QE (unit 0 to 1)
#     assumes that the wavelegth is in  nm"""
#     QE = spectral_response * wavelength / 1239.8
#     return QE


def implied_carrier(V: float, N: float, ni: float = 8697277437.298948, T: float = 298.15, n: int = 1) -> float:
    """
    Calculate excess carrier concentration.
    EQ: delta_n = (-N + sqrt(N^2 + 4 * ni^2 * exp(V / (n * k_B * T)))) / 2
    
    Args:
        V (float): Voltage (V).
        N (float): Doping concentration (cm⁻³).
        ni (float, optional): Intrinsic carrier concentration (cm⁻³). Default is 8697277437.298948.
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Excess carrier concentration (cm⁻³).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (-N + sp.sqrt(N**2 + 4 * ni**2 * sp.exp(V / (n * k_B * T)))) / 2
    if isinstance(res, sp.Number):
        return float(res)
    return res


# %% Current
def J0_layer(W: float, N: float, D: float, L: float, S: float, ni: float = 8697277437.298948) -> float:
    """
    Calculate saturation current density for the narrow case.
    EQ: J0 = q * ni^2 * F * D / (L * N)
    
    Args:
        W (float): Layer thickness (cm).
        N (float): Doping concentration (cm⁻³).
        D (float): Diffusivity (cm²/s).
        L (float): Diffusion length (cm).
        S (float): Surface recombination velocity (cm/s).
        ni (float, optional): Intrinsic carrier concentration (cm⁻³). Default is 8697277437.298948.
    
    Returns:
        float: Saturation current density (A/cm²).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))

    F = (S * sp.cosh(W / L) + D / L * sp.sinh(W * L)) / (
        D / L * sp.cosh(W * L) + S * sp.sinh(W / L)
    )
    res = q * ni**2 * F * D / (L * N)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def J0_factor(W: float, N: float, D: float, L: float, S: float, ni: float = 8697277437.298948) -> float:
    """
    Calculate saturation current density factor for the narrow case.
    EQ: F = (S * cosh(W / L) + D / L * sinh(W * L)) / (D / L * cosh(W * L) + S * sinh(W / L))
    
    Args:
        W (float): Layer thickness (cm).
        N (float): Doping concentration (cm⁻³).
        D (float): Diffusivity (cm²/s).
        L (float): Diffusion length (cm).
        S (float): Surface recombination velocity (cm/s).
        ni (float, optional): Intrinsic carrier concentration (cm⁻³). Default is 8697277437.298948.
    
    Returns:
        float: Saturation current density factor.
    """
    res = (S * sp.cosh(W / L) + D / L * sp.sinh(W * L)) / (
        D / L * sp.cosh(W * L) + S * sp.sinh(W / L)
    )
    if isinstance(res, sp.Number):
        return float(res)
    return res


def J0(ni: float, We: float, Ne: float, De: float, Le: float, Se: float, Nb: float, Wb: float, Db: float, Lb: float, Sb: float) -> float:
    """
    Calculate dark saturation current under the narrow base diode condition (L > W).
    EQ: J0 = q * ni^2 * (Fe * De / (Le * Ne) + Fb * Db / (Lb * Nb))
    
    Args:
        ni (float): Intrinsic carrier concentration (cm⁻³).
        We (float): Emitter thickness (cm).
        Ne (float): Emitter doping concentration (cm⁻³).
        De (float): Emitter diffusivity (cm²/s).
        Le (float): Emitter diffusion length (cm).
        Se (float): Emitter surface recombination velocity (cm/s).
        Nb (float): Base doping concentration (cm⁻³).
        Wb (float): Base width (cm).
        Db (float): Base diffusivity (cm²/s).
        Lb (float): Base diffusion length (cm).
        Sb (float): Base surface recombination velocity (cm/s).
    
    Returns:
        float: Dark saturation current (A/cm²).
    
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    Fe = (Se * sp.cosh(We / Le) + De / Le * sp.sinh(We * Le)) / (
        De / Le * sp.cosh(We * Le) + Se * sp.sinh(We / Le)
    )
    Fb = (Sb * sp.cosh(Wb / Lb) + Db / Lb * sp.sinh(Wb * Lb)) / (
        Db / Lb * sp.cosh(Wb * Lb) + Sb * sp.sinh(Wb / Lb)
    )
    res = q * ni**2 * (Fe * De / (Le * Ne) + Fb * Db / (Lb * Nb))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def current2gen(curr: float) -> float:
    """
    Convert current to generation rate.
    EQ: G = curr / q
    
    Args:
        curr (float): Current (A).
    
    Returns:
        float: Generation rate (eh pairs/s).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    res = curr / q
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_diode(V: float, I0: float, T: float = 298.15, n: int = 1) -> float:
    """
    Calculate current in an ideal diode.
    EQ: I = I0 * exp(V / (n * k_B * T) - 1)
    
    Args:
        V (float): Voltage (V).
        I0 (float): Saturation current (A).
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Current (A).
    
    Notes:
        For current density, I0 is in A/cm² and current density is returned.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = I0 * sp.exp(V / (n * k_B * T) - 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell(V: float, IL: float, I0: float, T: float = 298.15, n: int = 1) -> float:
    """
    Calculate current of a solar cell.
    EQ: I = IL - I0 * exp(V / (n * k_B * T))
    
    Args:
        V (float): Voltage (V).
        IL (float): Light generated current (A).
        I0 (float): Saturation current (A).
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Current (A).
    
    Notes:
        Also works for J0.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = IL - I0 * sp.exp(V / (n * k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell_DD(V: float, IL: float, I01: float, n1: int, I02: float, n2: int, T: float = 298.15) -> float:
    """
    Calculate current of a solar cell with double diode model.
    EQ: I = IL - I01 * (exp(V / (n1 * k_B * T)) - 1) - I02 * (exp(V / (n2 * k_B * T)) - 1)
    
    Args:
        V (float): Voltage (V).
        IL (float): Light generated current (A).
        I01 (float): Saturation current for first diode (A).
        n1 (int): Ideality factor for first diode.
        I02 (float): Saturation current for second diode (A).
        n2 (int): Ideality factor for second diode.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Current (A).

    Notes:
        Also works for J0.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    res = (
        IL
        - I01 * (sp.exp(V / (n1 * k_B * T)) - 1)
        - I02 * (sp.exp(V / (n2 * k_B * T)) - 1)
    )
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell_Rseries(V: float, Voc: float, Vmp: float, IL: float, I0: float, Imp: float) -> float:
    """
    Calculate current of a solar cell with series resistance.
    EQ: I = IL - C1 * exp(-Voc / C2) * (exp(V / C2) - 1)
    
    Args:
        V (float): Voltage (V).
        Voc (float): Open circuit voltage (V).
        Vmp (float): Voltage at maximum power point (V).
        IL (float): Light generated current (A).
        I0 (float): Saturation current (A).
        Imp (float): Current at maximum power point (A).
    
    Returns:
        float: Current (A).
    
    Notes:
        Also works for J0.
    """
    C1 = IL
    C2 = (Vmp - Voc) / (sp.log(1 - Imp / IL))

    res = IL - C1 * sp.exp(-1 * Voc / C2) * (sp.exp(V / C2) - 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def I_cell_Rshunt(V: float, IL: float, I0: float, Rshunt: float, T: float = 298.15, n: int = 1) -> float:
    """
    Calculate current of a solar cell with shunt resistance.
    EQ: I = IL - I0 * exp(V / (n * k_B * T)) - V / Rshunt
    
    Args:
        V (float): Voltage (V).
        IL (float): Light generated current (A).
        I0 (float): Saturation current (A).
        Rshunt (float): Shunt resistance (ohms).
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Current (A).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = IL - I0 * sp.exp(V / (n * k_B * T)) - V / Rshunt
    if isinstance(res, sp.Number):
        return float(res)
    return res


# %% Voltage
def impliedV(delta_n: float, N: float, T: float = 298.15, n: int = 1) -> float:
    """
    Calculate implied voltage.
    EQ: V = (n * k_B * T) * log((delta_n + N) * delta_n / ni^2)
    
    Args:
        delta_n (float): Excess carrier concentration (cm⁻³).
        N (float): Doping concentration (cm⁻³).
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Voltage (V).
    
    Notes:
        Implied voltage is often used to convert the carrier concentration in a lifetime tester to voltage.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (n * k_B * T) * sp.log((delta_n + N) * delta_n / ni_Si(T) ** 2)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def V_Rseries(voltage: float, curr: float, Rs: float) -> float:
    """
    Calculate voltage of a solar cell under the effect of series resistance.
    EQ: V = voltage - curr * Rs
    
    Args:
        voltage (float): Voltage (V).
        curr (float): Current (A).
        Rs (float): Series resistance (ohms).
    
    Returns:
        float: Voltage (V).
    """
    return voltage - curr * Rs


def Voc(IL: float, I0: float, T: float = 298.15, n: int = 1) -> float:
    """
    Calculate open circuit voltage.
    EQ: Voc = (n * k_B * T) * log(IL / I0 + 1)
    
    Args:
        IL (float): Light generated current (A).
        I0 (float): Saturation current (A).
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Open circuit voltage (V).
    
    Notes:
        IL and I0 must be in the same units.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (n * k_B * T) * sp.log(IL / I0 + 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def V_cell(curr: float, IL: float, I0: float, T: float = 298.15, n: int = 1) -> float:
    """
    Calculate voltage of a solar cell.
    EQ: V = (n * k_B * T) * log((IL - curr) / I0 + 1)
    
    Args:
        curr (float): Current (A).
        IL (float): Light generated current (A).
        I0 (float): Saturation current (A).
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Voltage (V).
    
    Notes:
        For current density, I0 is in A/cm² and current density is returned.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = (n * k_B * T) * sp.log((IL - curr) / I0 + 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


# %% Cell Resistances
def emitter_resistance(Rsheet, Sf):
    """
    Calculate contribution of the emitter to cell series resistance.
    EQ: 
    
    Args:
        Rsheet (float): Emitter sheet resistivity (ohm/sq).
        Sf (float): Finger spacing (cm).
    
    Returns:
        float: Series resistance (ohm·cm²).
    """
    return Rsheet * (Sf**2) / 12


def base_resistance(H, Nb, dopant="B"):
    """
    Calculate contribution of the base to cell series resistance.
    EQ: 
    
    Args:
        H (float): Base thickness (cm).
        Nb (float): Base doping concentration (cm⁻³).
        dopant (str, optional): Dopant type. Default is "B".
    
    Returns:
        float: Series resistance (ohm·cm²).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
    res = (1 / (q * mobility_generic(Nb, dopant) * Nb)) * H
    if isinstance(res, sp.Number):
        return float(res)
    return res


def finger_resistance(Rfinger, Sf, L, wf, df):
    """
    Calculate contribution of the finger to cell series resistance.
    EQ: 
    
    Args:
        Rfinger (float): Finger resistivity (ohm·cm).
        Sf (float): Finger spacing (cm).
        L (float): Finger length (cm).
        wf (float): Finger width (cm).
        df (float): Finger depth (cm).
    
    Returns:
        float: Series resistance (ohm·cm²).
    """
    return Rfinger * Sf * L**2 / (3 * wf * df)


def finger_resistivity(L, Jmp, Sf, resistivity, wf, df, Vmp):
    """
    Calculate fractional resistivity power loss in a finger.
    EQ: 
    
    Args:
        L (float): Finger length (cm).
        Jmp (float): Current density at maximum power point (A/cm²).
        Sf (float): Finger spacing (cm).
        resistivity (float): Finger resistivity (ohm·cm).
        wf (float): Finger width (cm).
        df (float): Finger depth (cm).
        Vmp (float): Voltage at maximum power point (V).
    
    Returns:
        float: Fractional resistivity power loss (%).
    """
    return (L**2 * Jmp * Sf * resistivity) / (3 * wf * df * Vmp) * 100.0


def finger_sheet(Sf, Jmp, Rsheet, Vmp):
    """
    Calculate fractional power loss due to finger sheet resistance.
    EQ: 
    
    Args:
        Sf (float): Finger spacing (cm).
        Jmp (float): Current density at maximum power point (A/cm²).
        Rsheet (float): Sheet resistivity (ohm/sq).
        Vmp (float): Voltage at maximum power point (V).
    
    Returns:
        float: Fractional power loss (%).
    """
    return (Sf**2 * Jmp * Rsheet) / (12 * Vmp) * 100.0


def busbar_resistance(Rbus, W, Z, wb, db, m):
    """
    Calculate contribution of the busbar to cell series resistance.
    EQ: 
    
    Args:
        Rbus (float): Busbar resistivity (ohm·cm).
        W (float): Busbar width (cm).
        Z (float): Busbar length (cm).
        wb (float): Busbar width (cm).
        db (float): Busbar depth (cm).
        m (int): Number of busbars.
    
    Returns:
        float: Series resistance (ohm·cm²).
    """
    return Rbus * W * Z**2 / (3 * wb * db * m)


def IBC_metal_resistance(Rmetal, W, Z, wfn, wfp, df, Sf):
    """
    Calculate metal resistance of the metal contacts on the back of an IBC cell.
    EQ: 
    
    Args:
        Rmetal (float): Metal resistivity (ohm·cm).
        W (float): Cell width (cm).
        Z (float): Cell length (cm).
        wfn (float): Width of n-type fingers (cm).
        wfp (float): Width of p-type fingers (cm).
        df (float): Finger depth (cm).
        Sf (float): Finger spacing (cm).
    
    Returns:
        float: Metal resistance (ohm·cm²).
    """
    unit = np.floor(Z / (wfn + Sf + wfp + Sf))
    #    center = unit*wfn*(W-.0125*2) + unit*wfp*(W-.0125*2)
    #    edge = Z*.01*2 + unit*wfn*.0025 + unit*wfp*.0025
    return Rmetal * W * Z**2 / (3 * wfn * df * unit) + Rmetal * W * Z**2 / (
        3 * wfp * df * unit
    )


# %% Cell Evaluations
def cell_params(V, curr):
    """
    Calculate key parameters of a solar cell IV curve.
    EQ: 
    
    Args:
        V (numpy.array): Voltage array (V).
        curr (numpy.array): Current array (A).
    
    Returns:
        tuple: Voc (V), Isc (A), FF, Vmp (V), Imp (A), Pmp (W).
    
    Notes:
        If curr is in (A/cm²) then Isc will be Jsc and Imp will be Jmp.
        No attempt is made to fit the fill factor.
    """
    Voc = np.interp(0, -curr, V)
    Isc = np.interp(0, V, curr)
    idx = np.argmax(V * curr)
    Vmp = V[idx]
    Imp = curr[idx]
    FF = Vmp * Imp / (Voc * Isc)
    return Voc, Isc, FF, Vmp, Imp, Vmp * Imp


def efficiency(Voc, Isc, FF, A=1):
    """
    Calculate efficiency of a solar cell.
    EQ: 
    
    Args:
        Voc (float): Open circuit voltage (V).
        Isc (float): Short circuit current (A).
        FF (float): Fill factor.
        A (float, optional): Area (cm²). Default is 1.
    
    Returns:
        float: Efficiency (%).
    
    Notes:
        Also works for Jsc since area of 1 is assumed.
    """
    return 1000 * Voc * Isc * FF / A


def finger_shading(wf, Sf):
    """
    Calculate fractional power loss due to finger shading.
    EQ: 
    
    Args:
        wf (float): Finger width (cm).
        Sf (float): Finger spacing (cm).
    
    Returns:
        float: Fractional power loss (%).
    """
    return (wf / Sf) * 100.0


def finger_total_loss(L, Jmp, Sf, resistivity, Rsheet, wf, df, Vmp):
    """
    Calculate total fractional power loss in a finger.
    EQ: 
    
    Args:
        L (float): Finger length (cm).
        Jmp (float): Current density at maximum power point (A/cm²).
        Sf (float): Finger spacing (cm).
        resistivity (float): Finger resistivity (ohm·cm).
        Rsheet (float): Sheet resistivity (ohm/sq).
        wf (float): Finger width (cm).
        df (float): Finger depth (cm).
        Vmp (float): Voltage at maximum power point (V).
    
    Returns:
        tuple: Total fractional power loss (%), resistivity loss (%), shading loss (%), sheet loss (%).
    """
    Presistivity = finger_resistivity(L, Jmp, Sf, resistivity, wf, df, Vmp)
    Pshading = finger_shading(wf, Sf)
    Psheet = finger_sheet(Sf, Jmp, Rsheet, Vmp)
    return Presistivity + Pshading + Psheet, Presistivity, Pshading, Psheet


def FF(Vmp, Imp, Voc, Isc):
    """
    Calculate fill factor of a solar cell.
    EQ: 
    
    Args:
        Vmp (float): Voltage at maximum power point (V).
        Imp (float): Current at maximum power point (A).
        Voc (float): Open circuit voltage (V).
        Isc (float): Short circuit current (A).
    
    Returns:
        float: Fill factor.
    """
    return (Vmp * Imp) / (Voc * Isc)


def FF_ideal(Voc, ideality=1, T=298.15):
    """
    Calculate ideal fill factor.
    EQ: 
    
    Args:
        Voc (float): Open circuit voltage (V).
        ideality (int, optional): Ideality factor. Default is 1.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Ideal fill factor.
    """
    voc = normalised_Voc(Voc, ideality, T)
    res = (voc - sp.log(voc + 0.72)) / (voc + 1)
    if isinstance(res, sp.Number):
        return float(res)
    return res


def normalised_Voc(Voc, ideality, T=298.15, n=1):
    """
    Calculate normalised open circuit voltage.
    EQ: 
    
    Args:
        Voc (float): Open circuit voltage (V).
        ideality (int): Ideality factor.
        T (float, optional): Temperature (K). Default is 298.15.
        n (int, optional): Ideality factor. Default is 1.
    
    Returns:
        float: Normalised open circuit voltage.
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)
    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))
    res = Voc / (ideality * (n * k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def FF_Rs(Voc, Isc, Rseries, ideality=1, T=298.15):
    """
    Calculate fill factor with series resistance.
    EQ: 
    
    Args:
        Voc (float): Open circuit voltage (V).
        Isc (float): Short circuit current (A).
        Rseries (float): Series resistance (ohms).
        ideality (int, optional): Ideality factor. Default is 1.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Fill factor.
    """
    RCH = Voc / Isc
    rs = Rseries / RCH
    FF0 = FF_ideal(Voc, ideality, T)
    FF = FF0 * (1 - 1.1 * rs) + (rs**2 / 5.4)
    return FF


def FF_Rsh(Voc, Isc, Rshunt, ideality=1, T=298.15):
    """
    Calculate fill factor with shunt resistance.
    EQ: 
    
    Args:
        Voc (float): Open circuit voltage (V).
        Isc (float): Short circuit current (A).
        Rshunt (float): Shunt resistance (ohms).
        ideality (int, optional): Ideality factor. Default is 1.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Fill factor.
    """
    voc = normalised_Voc(Voc, ideality, T)
    RCH = Voc / Isc
    rsh = Rshunt / RCH
    FF0 = FF_ideal(Voc, ideality, T)
    FF = FF0 * (1 - ((voc + 0.7) * FF0) / (voc * rsh))
    return FF


def FF_RsRsh(Voc, Isc, Rseries, Rshunt, ideality=1, T=298.15):
    """
    Calculate fill factor with series and shunt resistance.
    EQ: 
    
    Args:
        Voc (float): Open circuit voltage (V).
        Isc (float): Short circuit current (A).
        Rseries (float): Series resistance (ohms).
        Rshunt (float): Shunt resistance (ohms).
        ideality (int, optional): Ideality factor. Default is 1.
        T (float, optional): Temperature (K). Default is 298.15.
    
    Returns:
        float: Fill factor.
    """
    voc = normalised_Voc(Voc, ideality, T)
    RCH = Voc / Isc
    rsh = Rshunt / RCH
    # FF0 = FF_ideal(Voc, ideality, T)
    FFRs = FF_Rs(Voc, Isc, Rseries, ideality=1, T=298.15)
    FFRsRsh = FFRs * (1 - ((voc + 0.7) * FFRs) / (voc * rsh))
    return FFRsRsh


# %% silicon material properties
# silicon material properties
def optical_properties(fname):
    """
    Get optical properties of a material.
    EQ: 
    
    Args:
        fname (str): File name containing optical properties.
    
    Returns:
        tuple: 
            Wavelength (nm), 
            absorption coefficient (/cm), 
            real refractive index, 
            imaginary refractive index.
    
    Notes:
        If no file is given, silicon is used.

    if so file is given then silicon is used
    Eg: wavelength, abs_coeff, n, KB_J = optical_properties()
    """
    # if fname is None:
    #     package_path = os.path.dirname(os.path.abspath(__file__))
    #     fname = os.path.join(package_path, "silicon_optical_properties.txt")
    wavelength, abs_coeff, nd, kd = np.loadtxt(fname, skiprows=1, unpack=True)
    return wavelength, abs_coeff, nd, kd


# processing
def phos_active(T):
    """
    Calculate active limit of phosphorous in silicon.
    EQ: 
    
    Args:
        T (float): Temperature (K).
    
    Returns:
        float: Active limit of phosphorous (cm⁻³).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    res = 1.3e22 * sp.exp(-0.37 / (k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res


def phos_solubility(T):
    """
    Calculate solubility limit of phosphorous in silicon.
    EQ: 
    
    Args:
        T (float): Temperature (K).
    
    Returns:
        float: Solubility limit of phosphorous (cm⁻³).
    """
    arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)

    k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

    res = 2.45e23 * sp.exp(-0.62 / (k_B * T))
    if isinstance(res, sp.Number):
        return float(res)
    return res

    # # modules(Pedro)
    # def read_cell_info(selected, path=None, file=None):
    #     # TODO Rework with find_path
    #     if path is None:
    #         path = os.sep.join(("work", "Data"))
    #     package_path = pathify(path)
    #     if file is None:
    #         fname = os.path.join(package_path, "cell_info.txt")
    #     else:
    #         fname = os.path.join(package_path, file)

    #     with open(fname, "r") as f:
    #         for line in f:
    #             col1, col2, col3, col4 = line.split()
    #             if col1 == selected:
    #                 semicondutor = col1
    #                 J_SC = float(col2)
    #                 V_OC = float(col3)
    #                 J_0 = float(col4)
    #     return semicondutor, J_SC, V_OC, J_0

    # def module_current(M, N, T, material):
    #     arg_in = vars().copy()
    w_units = has_units(arg_in)
    symbolic = all_symbols(arg_in)


#     q = get_const("elementary_charge", *([True] if symbolic else [w_units, ["C"]]))
#     k_B = get_const("boltzmann", *([True] if symbolic else [w_units, ["eV", "K"]]))

#     semicondutor, J_SC, V_OC, J_0 = read_cell_info(material)
#     I_0 = J_0 * 15.6 * 15.6
#     I_L = J_SC * 15.6 * 15.6
#     V_T = V_OC * N
#     res = M * (I_L - I_0 * sp.exp((q * V_T / N) / (k_B * T) - 1))

#     if isinstance(res, sp.Number):
#         return float(res)
#     return res
