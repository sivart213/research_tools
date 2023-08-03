# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:54:59 2022.

@author: j2cle
"""

from .functions import (
    closest,
    Complexer,
    convert_prefix,
    convert_temp,
    convert_val,
    cost_base10,
    cost_basic,
    cost_log,
    cost_sqr,
    create_function,
    curve_fit_wrap,
    dict_df,
    dict_flat,
    dict_key_sep,
    dict_search,
    extract_arguments,
    find_nearest,
    function_to_expr,
    gen_bnds,
    gen_mask,
    get_const,
    has_arrays,
    has_symbols,
    all_symbols,
    pick_math_module,
    has_units,
    myprint,
    myround,
    parse_unit,
    parse_constant,
    precise_round,
    sample_array,
    sci_note,
    sig_figs_ceil,
    sig_figs_round,
    solve_for_variable,
    bode,
    bode2,
    lineplot_slider,
    map_plt,
    nyquist,
    nyquist2,
    scatter,
    f_find,
    get_config,
    load,
    load_hdf,
    p_find,
    pathify,
    pathlib_mk,
    PickleJar,
    save,
    slugify,
)


from .equations import (
    arc,
    arccosd,
    arcsind,
    arrh,
    cosd,
    CtoF,
    CtoK,
    CtoR,
    erf,
    erfc,
    erfinv,
    erfcinv,
    FtoC,
    FtoK,
    FtoR,
    inv_sum_invs,
    KtoC,
    KtoF,
    KtoR,
    line,
    polynomial,
    RtoC,
    RtoF,
    RtoK,
    sind,
    sphere_area,
    sphere_vol,
    Statistics,
    tand,
    bandgap_paessler,
    bandgap_schenk,
    bjerrum_length,
    capacitance,
    characteristic_length,
    conductivity,
    debye_length,
    depletion_region,
    diffusion_length,
    equilibrium_carrier,
    lifetime,
    lifetime_auger,
    lifetime_bulk,
    lifetime_eff,
    lifetime_minority,
    lifetime_SRH,
    mobility_diffusion,
    mobility_generic,
    mobility_klassen,
    mobility_masetti,
    mobility_thurber,
    nernst_planck_analytic_sol,
    nernst_planck_fundamental_sol,
    ni_eff,
    ni_Si,
    ohms_law,
    probability_bose_einstein,
    probability_fermi_dirac,
    probability_maxwell_boltzmann,
    resistance,
    resistivity,
    resistivity_Si_n,
    resistivity_Si_p,
    screened_permitivity,
    sheet_resistivity,
    U_auger_richter,
    U_low_doping,
    U_radiative,
    U_radiative_alt,
    U_SRH,
    U_surface,
    v_thermal,
    voltage_divider,
    base_resistance,
    busbar_resistance,
    cell_params,
    current2gen,
    efficiency,
    emitter_resistance,
    FF,
    FF_ideal,
    FF_Rs,
    FF_Rsh,
    FF_RsRsh,
    finger_resistance,
    finger_resistivity,
    finger_shading,
    finger_sheet,
    finger_total_loss,
    I_cell,
    I_cell_DD,
    I_cell_Rseries,
    I_cell_Rshunt,
    I_diode,
    IBC_metal_resistance,
    implied_carrier,
    impliedV,
    IQE,
    IQE_base,
    IQE_bulk,
    IQE_bulk2,
    IQE_depletion,
    IQE_emitter,
    IQE_IBC_emitter1,
    IQE_IBC_emitter2,
    J0,
    J0_factor,
    J0_layer,
    normalised_Voc,
    optical_properties,
    phos_active,
    phos_solubility,
    V_cell,
    V_Rseries,
    Voc,
)

__all__ = [
    "closest",
    "Complexer",
    "convert_prefix",
    "convert_temp",
    "convert_val",
    "cost_base10",
    "cost_basic",
    "cost_log",
    "cost_sqr",
    "create_function",
    "curve_fit_wrap",
    "dict_df",
    "dict_flat",
    "dict_key_sep",
    "dict_search",
    "extract_arguments",
    "find_nearest",
    "function_to_expr",
    "gen_bnds",
    "gen_mask",
    "get_const",
    "has_arrays",
    "has_symbols",
    "all_symbols",
    "pick_math_module",
    "has_units",
    "myprint",
    "myround",
    "parse_unit",
    "parse_constant",
    "precise_round",
    "sample_array",
    "sci_note",
    "sig_figs_ceil",
    "sig_figs_round",
    "solve_for_variable",
    "bode",
    "bode2",
    "lineplot_slider",
    "map_plt",
    "nyquist",
    "nyquist2",
    "scatter",
    "f_find",
    "get_config",
    "load",
    "load_hdf",
    "p_find",
    "pathify",
    "pathlib_mk",
    "PickleJar",
    "save",
    "slugify",
    "arc",
    "arccosd",
    "arcsind",
    "arrh",
    "cosd",
    "CtoF",
    "CtoK",
    "CtoR",
    "erf",
    "erfc",
    "erfinv",
    "erfcinv",
    "FtoC",
    "FtoK",
    "FtoR",
    "inv_sum_invs",
    "KtoC",
    "KtoF",
    "KtoR",
    "line",
    "polynomial",
    "RtoC",
    "RtoF",
    "RtoK",
    "sind",
    "sphere_area",
    "sphere_vol",
    "Statistics",
    "tand",
    "bandgap_paessler",
    "bandgap_schenk",
    "bjerrum_length",
    "capacitance",
    "characteristic_length",
    "conductivity",
    "debye_length",
    "depletion_region",
    "diffusion_length",
    "equilibrium_carrier",
    "lifetime",
    "lifetime_auger",
    "lifetime_bulk",
    "lifetime_eff",
    "lifetime_minority",
    "lifetime_SRH",
    "mobility_diffusion",
    "mobility_generic",
    "mobility_klassen",
    "mobility_masetti",
    "mobility_thurber",
    "nernst_planck_analytic_sol",
    "nernst_planck_fundamental_sol",
    "ni_eff",
    "ni_Si",
    "ohms_law",
    "probability_bose_einstein",
    "probability_fermi_dirac",
    "probability_maxwell_boltzmann",
    "resistance",
    "resistivity",
    "resistivity_Si_n",
    "resistivity_Si_p",
    "screened_permitivity",
    "sheet_resistivity",
    "U_auger_richter",
    "U_low_doping",
    "U_radiative",
    "U_radiative_alt",
    "U_SRH",
    "U_surface",
    "v_thermal",
    "voltage_divider",
    "base_resistance",
    "busbar_resistance",
    "cell_params",
    "current2gen",
    "efficiency",
    "emitter_resistance",
    "FF",
    "FF_ideal",
    "FF_Rs",
    "FF_Rsh",
    "FF_RsRsh",
    "finger_resistance",
    "finger_resistivity",
    "finger_shading",
    "finger_sheet",
    "finger_total_loss",
    "I_cell",
    "I_cell_DD",
    "I_cell_Rseries",
    "I_cell_Rshunt",
    "I_diode",
    "IBC_metal_resistance",
    "implied_carrier",
    "impliedV",
    "IQE",
    "IQE_base",
    "IQE_bulk",
    "IQE_bulk2",
    "IQE_depletion",
    "IQE_emitter",
    "IQE_IBC_emitter1",
    "IQE_IBC_emitter2",
    "J0",
    "J0_factor",
    "J0_layer",
    "normalised_Voc",
    "optical_properties",
    "phos_active",
    "phos_solubility",
    "V_cell",
    "V_Rseries",
    "Voc",
]