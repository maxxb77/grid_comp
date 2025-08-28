#!/usr/bin/env python3

# !!!! note - in this version, the steel facilities CF and electricity
# !!!! costs also depend on retail adders and avg_lcoe
# !!!! see lines

import argparse
import os
from itertools import product
from threading import Thread

import numpy as np
import pandas as pd

root_dir = "/Users/max/Documents/GitHub/"


# ----------------------------- constants ----------------------------- #

CDT_FACTOR = 1.17

inflate_2004 = 1.703
inflate_2018 = 1.271
inflate_2023 = 1.055
#ongrid capacity factor
CAP_FACTOR = 0.85
# no longer used.. cancelled out
CAP_FACTOR_OFFGRID = 0.441207
EQUITY_SHARE = 0.4
INT_DEBT = 0.05
TAX_RATE = 0.25
RROE_REAL = 0.08
FAC_LIFETIME = 30
INFLATE = 1.057  # retained for parity
EMIT_COAL = 0.094
EMIT_DDFO = 0.074
EMIT_NG = 0.053

LCOE_YEAR = 2030
CHOSEN_LCOE_INPUT = "der_mean"  # original choice
CHOSEN_LCOE = "lcoe_min"        # choose "lcoe_min" or "lcoe_blend" below

CAMBIUM_CASES = [
    "HighNGPrice", "LowNGPrice", "MidCase", "HighRECost",
    "LowRECost", "HighRECost_LowNGPrice", "LowRECost_HighNGPrice"
]

H2_TECHS_TEXAS = ["h2_pem_electrol_gh500", "h2_pem_electrol_gh1000", "h2_pem_electrol_gh2000"]
TECH_KEEP = [
    "h2_pem_electrol", "h2_soec_electrol", "h2_smr", "h2_smr_ccs",
    "h2_pem_electrol_gh500", "h2_pem_electrol_gh1000", "h2_pem_electrol_gh2000"
]

RETAIL_ADDERS = [0, 15, 30, 45]


# ----------------------------- small helpers ----------------------------- #

def format_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = df2.columns.str.replace(" ", "_")
    return df2


def to_lower_nospace(series: pd.Series) -> pd.Series:
    return series.str.lower().str.replace(" ", "", regex=False)


def calc_wacc(equity_share: float = EQUITY_SHARE,
              int_debt: float = INT_DEBT,
              tax_rate: float = TAX_RATE,
              rroe_real: float = RROE_REAL) -> float:
    return (1 - equity_share) * int_debt * (1 - tax_rate) + equity_share * rroe_real


def calc_crf(rate: float, years: int) -> float:
    return (rate * (1 + rate) ** years) / ((1 + rate) ** years - 1)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------- main pipeline ----------------------------- #

def main(root_dir):
    # paths
    lcoh_dir = os.path.join(root_dir, "grid_comp")

    # core inputs
    cf_char = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "char_cf.csv"))
    fuel_prices_state = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "fuel_prices_state.csv"))
    state_name_abb = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "state_name_abb.csv"))
    state_name_abb["state_name"] = to_lower_nospace(state_name_abb["state_name"])
    state_name_abb["state_abb"] = to_lower_nospace(state_name_abb["state_abb"])

    # preprocess
    cf_char = format_columns(cf_char).fillna(0)
    cf_char = cf_char[cf_char["HYD"] != "fraction [rate]"]  # remove units row

    h2_char = cf_char.copy()
    h2_char["HYD"] = h2_char["HYD"].astype(float)
    h2_char = h2_char[h2_char["HYD"] > 0]
    h2_char = h2_char[h2_char["id"] == "new"]

    remove_columns_h2 = [
        "id", "r", "lat", "lon", "int_elec_self", "int_h2", "int_h2_mark", "int_h2_self",
        "int_co2", "int_co2_mark", "int_co2_self", "ee", "cod_dec", "cap_dec", "outage_rate",
        "min_cap", "int_met_coal_feed", "int_ddfo", "co2_rate_comb", "co2_rate_proc", "GAS",
        "JFL", "DDFO", "ETH", "HYD", "COKE"
    ]
    h2_char = h2_char.drop(remove_columns_h2, axis=1)
    h2_char_orig = h2_char.copy()

    # ------------------ H2 permutations (fuel grids) ------------------ #

    gas_prices = list(range(0, 11, 2))     # $/MMBtu
    ele_prices = list(range(0, 101, 10))   # $/MWh

    co2_ppm = 805
    co2_mpm = int((round(co2_ppm / 2200, 5) * 1e3))
    co2_ele = list(range(0, 2 * co2_mpm, 100))  # electricity emissions ladder (lbs->scaled units per original)

    co2_ng = [0.053, 0.064, 0.075, 0.082, 0.086, 0.111, 0.140]
    co2_tax = [0]
    co2_tns = [5, 15, 25]

    fuel_comb = pd.DataFrame(
        list(product(gas_prices, ele_prices, co2_ele, co2_ng, co2_tax, co2_tns)),
        columns=["gas_price", "ele_price", "co2_ele", "co2_ng", "co2_tax", "co2_tns"]
    )

    h2_perm = pd.DataFrame()
    for _, row in fuel_comb.iterrows():
        temp = h2_char.copy()
        temp["gas_price"] = row["gas_price"]
        temp["ele_price"] = row["ele_price"]
        temp["co2_ele"] = row["co2_ele"] / 1e3
        temp["co2_ng"] = row["co2_ng"]
        temp["co2_tax"] = row["co2_tax"]
        temp["co2_tns"] = row["co2_tns"]
        h2_perm = pd.concat([h2_perm, temp], ignore_index=True)

    # finance (same math as original)
    wacc_par = calc_wacc()
    crf_par = calc_crf(wacc_par, FAC_LIFETIME)

    h2_perm["cost_cap"] = h2_perm["cost_cap"].astype(float)
    h2_perm["LCOF_cap"] = h2_perm["cost_cap"] / 114.877 / 365 * CDT_FACTOR * crf_par / CAP_FACTOR
    h2_perm["LCOF_vom"] = h2_perm["cost_vom"].astype(float)
    h2_perm["emit_rate_comb"] = h2_perm["co2_ng"] * h2_perm["int_ng"].astype(float)
    h2_perm["emit_ele"] = h2_perm["co2_ele"] * h2_perm["int_elec"].astype(float)
    h2_perm["emit_rate_total"] = h2_perm["emit_rate_comb"] + h2_perm["emit_ele"]
    h2_perm["emit_captured"] = h2_perm["ccs_cap_rate_comb"].astype(float) * EMIT_NG * h2_perm["int_ng"].astype(float)
    h2_perm["LCOF_co2_tax_cost"] = h2_perm["co2_tax"] * h2_perm["emit_rate_total"]
    h2_perm["LCOF_co2_tns"] = h2_perm["co2_tns"] * h2_perm["emit_captured"]
    h2_perm["LCOF_energy_gas"] = h2_perm["int_ng"].astype(float) * h2_perm["gas_price"]
    h2_perm["LCOF_energy_elec"] = h2_perm["int_elec"].astype(float) * h2_perm["ele_price"]
    h2_perm["LCOF_fom"] = h2_perm["cost_fom_per_metric_ton"].astype(float) / 114.877

    components = [
        "LCOF_cap", "LCOF_vom", "LCOF_fom", "LCOF_co2_tax_cost",
        "LCOF_co2_tns", "LCOF_energy_gas", "LCOF_energy_elec"
    ]
    id_vars1 = [x for x in list(h2_perm.columns) if x not in components]
    h2_out = pd.melt(h2_perm, value_name="LCOF_Cost", value_vars=components, id_vars=id_vars1)
    h2_out.to_csv(os.path.join(lcoh_dir, "h2_perm.csv"), index=False)

    # ------------------ LCOH by state ------------------ #

    fuel_prices_state = fuel_prices_state.drop(["Region"], axis=1)
    h2_state = h2_char_orig.merge(fuel_prices_state, how="cross")

    h2_state["cost_cap"] = h2_state["cost_cap"].astype(float)
    h2_state["LCOF_cap"] = h2_state["cost_cap"] / 114.877 / 365 * CDT_FACTOR * crf_par / CAP_FACTOR
    h2_state["LCOF_vom"] = h2_state["cost_vom"].astype(float)
    h2_state["emit_captured"] = h2_state["ccs_cap_rate_comb"].astype(float) * EMIT_NG * h2_state["int_ng"].astype(float)
    h2_state["LCOF_co2_tns"] = inflate_2004 * h2_state["CCS2"].astype(float) * h2_state["emit_captured"]
    h2_state["LCOF_energy_gas"] = h2_state["int_ng"].astype(float) * h2_state["Gas"]
    h2_state["LCOF_energy_elec"] = 10 * h2_state["int_elec"].astype(float) * h2_state["ELE_IND"]  # cents/kWh -> $/MWh
    h2_state["LCOF_fom"] = h2_state["cost_fom_per_metric_ton"].astype(float) / 114.877

    components_state = ["LCOF_cap", "LCOF_vom", "LCOF_fom", "LCOF_co2_tns", "LCOF_energy_gas", "LCOF_energy_elec"]
    id_vars2 = [x for x in list(h2_state.columns) if x not in components_state]
    h2_state_out = pd.melt(h2_state, value_name="LCOF_Cost", value_vars=components_state, id_vars=id_vars2)
    h2_state_out.to_csv(os.path.join(lcoh_dir, "lcoh_state.csv"), index=False)

    # ------------------ H2 by county ------------------ #

    c2z_loc = os.path.join(lcoh_dir, "raw_data", "county2zone.csv")
    c2z = pd.read_csv(c2z_loc, names=["fips", "ba", "county_name", "state_abb"], header=0, dtype={"fips": str})

    slope_cols_new = [
        "county_name", "state_full", "state_code", "ele_tech", "year", "gid",
        "lcoe_mean", "lcoe_min", "lcoe_max", "lcoe_med", "sd_a", "sd_b", "sd_c", "der_mean"
    ]
    lcoe_county = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "slope_2024.csv"),
                              names=slope_cols_new, header=0, dtype={"fips": str})
    lcoe_county["fips"] = lcoe_county["gid"].str[1:3] + lcoe_county["gid"].str[4:7]

    # ensure wind rows exist by copying pv rows, set land-based-wind to 51.35 (average) when missing
    for i in lcoe_county["fips"].unique():
        mask = (lcoe_county["fips"] == i) & (lcoe_county["ele_tech"] == "land-based-wind")
        if len(lcoe_county[mask]) == 0:
            temp = lcoe_county[(lcoe_county["fips"] == i) & (lcoe_county["ele_tech"] == "pv")].copy()
            temp["ele_tech"] = "land-based-wind"
            temp[CHOSEN_LCOE_INPUT] = 51.35
            lcoe_county = pd.concat([lcoe_county, temp], ignore_index=True)

    # PTC inflation-adjusted adder
    ptc_adder = 27.5
    lcoe_county.loc[lcoe_county["ele_tech"] == "land-based-wind", CHOSEN_LCOE_INPUT] = \
        lcoe_county.loc[lcoe_county["ele_tech"] == "land-based-wind", CHOSEN_LCOE_INPUT] + ptc_adder
    lcoe_county.loc[lcoe_county["ele_tech"] == "pv", CHOSEN_LCOE_INPUT] = \
        lcoe_county.loc[lcoe_county["ele_tech"] == "pv", CHOSEN_LCOE_INPUT] + ptc_adder

    pd.DataFrame(lcoe_county).to_csv(os.path.join(lcoh_dir, "lcoe_plot.csv"), index=False)

    # filter to year and subset
    lcoe_county_sub = lcoe_county[lcoe_county["year"] == LCOE_YEAR]
    lcoe_county_sub = lcoe_county_sub[["fips", "ele_tech", CHOSEN_LCOE_INPUT]]

    remove_tech = [
        "btm", "battery", "coal", "commercial_pv", "fom", "gas-cc", "gas-ct",
        "geothermal", "residential_pv"
    ]
    lcoe_county_nobattery = lcoe_county_sub[~lcoe_county_sub["ele_tech"].isin(remove_tech)]
    lcoe_county_nobattery = lcoe_county_nobattery[lcoe_county_nobattery[CHOSEN_LCOE_INPUT] > 0]

    lcoe_county_sub2 = lcoe_county_sub[
        (lcoe_county_sub["ele_tech"] == "land-based-wind") | (lcoe_county_sub["ele_tech"] == "pv")
    ]
    lcoe_min = lcoe_county_nobattery.groupby("fips")[CHOSEN_LCOE_INPUT].min().reset_index()
    lcoe_min["ele_tech"] = "min_alltechs"
    lcoe_county_out = pd.concat([lcoe_min, lcoe_county_sub2]).drop_duplicates().reset_index(drop=True)

    county_wide = lcoe_county_out.pivot(index=["fips"], columns=["ele_tech"], values=CHOSEN_LCOE_INPUT).reset_index()
    county_wide.columns = ["fips", "lcoe_wind", "lcoe_min", "lcoe_pv"]

    county_wide = pd.merge(county_wide, c2z[["fips", "state_abb"]], on="fips")
    county_wide["state_abb"] = to_lower_nospace(county_wide["state_abb"])

    wind_share = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "wind_share.csv"),
                             names=["state_name", "share"], header=0)
    wind_share["state_name"] = to_lower_nospace(wind_share["state_name"])

    wind_share = pd.merge(wind_share, state_name_abb, on="state_name")
    county_wide = pd.merge(county_wide, wind_share[["state_abb", "share"]], on="state_abb", how="left")
    county_wide["lcoe_blend"] = county_wide["lcoe_wind"] * county_wide["share"] + \
        county_wide["lcoe_pv"] * (1 - county_wide["share"])

    # choose lcoe_min or lcoe_blend
    county_wide = pd.merge(county_wide, wind_share, how="left")

    fuel_prices_state2 = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "fuel_prices_state.csv"))
    fuel_prices_state2["state_name"] = to_lower_nospace(fuel_prices_state2["State"])
    fuel_prices_state2 = pd.merge(fuel_prices_state2, state_name_abb)
    fuel_prices_state2 = fuel_prices_state2[["state_abb", "ELE_IND", "Gas", "CCS2"]]
    fuel_prices_state2.columns = ["state_abb", "grid_ele_price", "gas_price", "ccs_cost"]

    county_wide = pd.merge(county_wide, fuel_prices_state2, on="state_abb")
    temp_wide = county_wide[["fips", "state_abb", CHOSEN_LCOE, "grid_ele_price", "gas_price", "ccs_cost"]].copy()
    temp_wide.columns = ["fips", "state_abb", "offgrid_ele_price", "grid_ele_price", "gas_price", "ccs_cost"]

    h2_county = h2_char_orig.copy()
    h2_county = h2_county[["pathway", "cost_cap", "cost_fom_per_metric_ton", "ccs_cap_rate_comb", "cost_vom", "int_elec", "int_ng"]]
    h2_county = h2_county.merge(county_wide, how="cross")

    h2_county["cost_cap"] = h2_county["cost_cap"].astype(float)
    h2_county["LCOF_cap"] = h2_county["cost_cap"] / 114.877 / 365 * CDT_FACTOR * crf_par / CAP_FACTOR
    h2_county["LCOF_vom"] = h2_county["cost_vom"].astype(float)
    h2_county["emit_captured"] = h2_county["ccs_cap_rate_comb"].astype(float) * EMIT_NG * h2_county["int_ng"].astype(float)
    h2_county["LCOF_co2_tns"] = h2_county["ccs_cost"].astype(float) * h2_county["emit_captured"]
    h2_county["LCOF_energy_gas"] = h2_county["int_ng"].astype(float) * h2_county["gas_price"]
    h2_county["LCOF_energy_elec_ongrid"] = 10 * h2_county["int_elec"].astype(float) * h2_county["grid_ele_price"]
    h2_county["LCOF_energy_elec_wind"] = h2_county["int_elec"].astype(float) * h2_county["lcoe_wind"]
    h2_county["LCOF_energy_elec_pv"] = h2_county["int_elec"].astype(float) * h2_county["lcoe_pv"]
    h2_county["LCOF_energy_elec_min"] = h2_county["int_elec"].astype(float) * h2_county["lcoe_min"] 
    h2_county["LCOF_energy_elec_blend"] = h2_county["int_elec"].astype(float) * h2_county["lcoe_blend"]
    h2_county["LCOF_fom"] = h2_county["cost_fom_per_metric_ton"].astype(float) / 114.877

    h2_transport_stor = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "h2_transport_and_storage_costs.csv"),
                                    names=["type", "t", "parameter", "value"], header=0)
    h2_stor = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "h2_storage_rb.csv"),
                          names=["type", "ba"], header=0)
    h2_stor_char = pd.merge(h2_stor, h2_transport_stor, how="left")
    h2_stor_char = h2_stor_char[h2_stor_char["parameter"].isin(["cost_cap", "fom"])]
    h2_stor_char["LCOS"] = 0
    h2_stor_char.loc[h2_stor_char["parameter"] == "cost_cap", "LCOS"] = (h2_stor_char["value"] / 8760 / 114.877) / 0.15
    h2_stor_char.loc[h2_stor_char["parameter"] == "fom", "LCOS"] = h2_stor_char["value"] / 1e4
    h2_stor_char = h2_stor_char[h2_stor_char["t"] == LCOE_YEAR]
    h2_stor_char = h2_stor_char[["ba", "parameter", "LCOS"]]
    h2_stor_char = h2_stor_char.pivot(index="ba", columns="parameter", values="LCOS").reset_index()
    h2_stor_char.columns = ["ba", "LCOS_cap", "LCOS_fom"]

    h2_county_withba = pd.merge(h2_county, c2z[["fips", "ba"]], on="fips")
    h2_county = h2_county_withba.merge(h2_stor_char, on="ba", how="left")

    components_county = [
        "LCOF_cap", "LCOF_vom", "LCOF_fom", "LCOF_co2_tns", "LCOF_energy_gas",
        "LCOF_energy_elec_ongrid", "LCOF_energy_elec_wind", "LCOF_energy_elec_pv",
        "LCOF_energy_elec_min", "LCOF_energy_elec_blend", "LCOS_cap", "LCOS_fom"
    ]
    id_vars_county = [x for x in list(h2_county.columns) if x not in components_county]
    h2_county_out = pd.melt(h2_county, value_name="LCOF_Cost", value_vars=components_county, id_vars=id_vars_county)

    ongrid_vars = ["LCOF_cap", "LCOF_vom", "LCOF_fom", "LCOF_co2_tns", "LCOF_energy_gas", "LCOF_energy_elec_ongrid", "LCOS_cap", "LCOS_fom"]
    offgrid_vars = ["LCOF_cap", "LCOF_vom", "LCOF_fom", "LCOF_co2_tns", "LCOF_energy_gas", "LCOF_energy_elec_min", "LCOS_cap", "LCOS_fom"]

    h2_county_out_offgrid = h2_county_out.copy()
    h2_county_out_offgrid["style"] = "offgrid"
    h2_county_out_offgrid = h2_county_out_offgrid[h2_county_out_offgrid["variable"].isin(offgrid_vars)]

    texas_cf = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "cf_texas.csv"))
    h2_county_out_offgrid.loc[h2_county_out_offgrid["variable"] == "LCOF_cap", "LCOF_Cost"] = \
        h2_county_out_offgrid.loc[h2_county_out_offgrid["variable"] == "LCOF_cap", "LCOF_Cost"] / inflate_2018

    h2_county_out_ongrid = h2_county_out.copy()
    h2_county_out_ongrid["style"] = "ongrid"
    h2_county_out_ongrid = h2_county_out_ongrid[h2_county_out_ongrid["variable"].isin(ongrid_vars)]

    h2_county_out_final = pd.concat([h2_county_out_ongrid, h2_county_out_offgrid], ignore_index=True)
    h2_county_out_final.to_csv(os.path.join(lcoh_dir, "county_h2_out.csv"), index=False)

    # Texas comparison
    h2_techs = H2_TECHS_TEXAS
    h2_county_comp = h2_county_out_final[h2_county_out_final["state_abb"] == "tx"].copy()
    h2_county_comp = h2_county_comp[h2_county_comp["pathway"].isin(h2_techs)]
    texas_cf = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "cf_texas.csv"))
    texas_cf.columns = ["cat", "fips", "t", "cf"]
    texas_cf = texas_cf[texas_cf["t"] == 2032]
    texas_cf = texas_cf[texas_cf["cat"] == "avg"]
    texas_cf["fips"] = texas_cf["fips"].str.replace("p", "")
    h2_county_comp = h2_county_comp.merge(texas_cf[["fips", "cf"]], how="left")
    mask_min = h2_county_comp["variable"] == "LCOF_energy_elec_min"
    h2_county_comp.loc[mask_min, "LCOF_Cost"] = (
        h2_county_comp.loc[mask_min, "LCOF_Cost"] / h2_county_comp.loc[mask_min, "cf"])
    h2_county_comp.to_csv(os.path.join(lcoh_dir, "texas_temp.csv"), index=False)

    # ------------------ LCOS by state / steel ------------------ #

    steel_county = temp_wide.copy()
    steel_county = steel_county.merge(c2z[["fips", "ba"]], how="left")

    ore_transport = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "finito_iron_ore_transport_cost.csv"), header=0)
    ore_transport.columns = ore_transport.columns.str.lower()
    ore_transport = ore_transport[["ba", "2030"]]
    ore_transport.columns = ["ba", "ore_transport"]
    steel_county = steel_county.merge(ore_transport, on="ba", how="left")

    finito_prices = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "fuelprices_finito.csv"))
    finito_prices["state_abb"] = to_lower_nospace(finito_prices["state_abb"])
    finito_prices = finito_prices[finito_prices["state_abb"] != "voluntary"]
    finito_prices = finito_prices[finito_prices["t"] == LCOE_YEAR]
    finito_prices = finito_prices[["ei", "state_abb", "cost"]]
    finito_prices = finito_prices.pivot(index="state_abb", columns="ei", values="cost").reset_index()

    steel_prices = steel_county.merge(finito_prices, on="state_abb", how="left")
    steel_prices.columns = steel_prices.columns.str.replace("int", "price")

    lcoh_tech = h2_county_out_final.groupby(["fips", "pathway", "style"])["LCOF_Cost"].sum().reset_index()
    steel_matrix = steel_prices.merge(lcoh_tech, on="fips", how="left")

    steel_char = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "char_ind.csv"))
    steel_char = steel_char[(steel_char["id"] == "new") & (steel_char["commodity"] == "steel")]

    drop_steel = [
        "id", "ba_zone", "commodity", "fac_id", "fac_zip", "lat", "lon", "cod", "ref_cap", "cap_dec", "prod_2018",
        "int_hgl", "int_opet", "int_ng_feed", "int_coal_feed", "int_coke_feed", "int_lighthgl_feed",
        "int_medhgl_feed", "int_heavypchem_feed", "int_m_limestone", "int_m_cullet", "int_m_silica", "int_m_soda_ash"
    ]
    steel_char = steel_char.drop(drop_steel, axis=1)
    steel_char_copy = steel_char.copy()

    wacc_par2 = calc_wacc()
    crf_par2 = calc_crf(wacc_par2, FAC_LIFETIME)

    steel_perm = steel_matrix.merge(steel_char, how="cross")
    steel_perm["lcos_coal"] = steel_perm["int_coal"].astype(float) * steel_perm["price_coal"].astype(float)
    steel_perm["lcos_met_coal"] = steel_perm["int_met_coal"].astype(float) * steel_perm["price_met_coal"].astype(float)
    steel_perm["lcos_coke"] = steel_perm["int_coke"].astype(float) * steel_perm["price_coke"].astype(float)
    steel_perm["lcos_ddfo"] = steel_perm["int_ddfo"].astype(float) * steel_perm["price_ddfo"].astype(float)
    steel_perm["lcos_ng"] = steel_perm["int_ng"].astype(float) * steel_perm["gas_price"].astype(float)
    steel_perm["lcos_elec"] = steel_perm["int_elec"].astype(float) * steel_perm["price_elec"].astype(float)
    steel_perm["lcos_rfo"] = steel_perm["int_rfo"].astype(float) * steel_perm["price_rfo"].astype(float)
    steel_perm["lcos_h2_feed"] = 114 * steel_perm["int_h2_feed"].astype(float) * steel_perm["LCOF_Cost"].astype(float)

    steel_perm["lcos_m_scrap"] = steel_perm["int_m_scrap"].astype(float) * 325
    steel_perm["lcos_m_ore"] = steel_perm["int_m_ore"].astype(float) * (111.06 + steel_perm["ore_transport"])
    steel_perm["lcos_vom"] = steel_perm["cost_vom"]
    steel_perm["lcos_cap"] = steel_perm["cost_cap"].astype(float) * CDT_FACTOR * crf_par2 / CAP_FACTOR
    steel_perm["lcos_fom"] = steel_perm["cost_fom"].astype(float) / CAP_FACTOR

    steel_perm["lcos_ccs"] = inflate_2004 * steel_perm["ccs_cost"].astype(float) * steel_perm["ccs_cap_rate_comb"].astype(float) * (
        (steel_perm["int_coal"].astype(float) * 0.095
         + steel_perm["int_ng"].astype(float) * 0.053
         + steel_perm["int_coke"].astype(float) * 0.113
         + steel_perm["int_ddfo"].astype(float) * 0.074
         + steel_perm["int_rfo"].astype(float) * 0.074
         + steel_perm["int_met_coal"].astype(float) * 0.093)
        + steel_perm["co2_rate_proc"].astype(float)
    )

    steel_char_cols = [
        "lcos_coal", "lcos_met_coal", "lcos_coke", "lcos_ddfo", "lcos_ng", "lcos_elec",
        "lcos_elec", "lcos_rfo", "lcos_h2_feed", "lcos_m_scrap", "lcos_m_ore",
        "lcos_vom", "lcos_cap", "lcos_fom", "lcos_ccs"
    ]
    steel_out = steel_perm[
        ["fips", "type", "pathway", "style"] + steel_char_cols[:-1]  # same selection as original
    ]
    steel_county_out = pd.melt(
        steel_perm,
        value_name="lcos",
        value_vars=steel_char_cols,
        id_vars=["fips", "type", "pathway", "style"]
    )
    steel_county_out.to_csv(os.path.join(lcoh_dir, "steel_county.csv"), index=False)

    # ------------------ Cambium loop ------------------ #

    cambium_year = 2030
    num_thread = 10
    lcoh_min_full = pd.DataFrame()
    steel_full = pd.DataFrame()
    cambium_min = pd.DataFrame()

    def run_region(ba_list, results, index, case):
        cam_out = pd.DataFrame()
        for r in ba_list:
            cam_in = pd.read_csv(
                os.path.join(lcoh_dir, "raw_data", "Cambium24", case, f"Cambium24_{case}_hourly_{r}_{cambium_year}.csv"),
                header=5
            )[["total_cost_enduse"]]
            cam_in["rank"] = cam_in.rank(method="first")
            for i in cam_in["rank"].unique():
                cam_temp = cam_in[cam_in["rank"] <= i]["total_cost_enduse"].mean()
                cam_stack = pd.DataFrame({"avg_lcoe": [cam_temp], "rank": [i], "r": [r]})
                cam_out = pd.concat([cam_out, cam_stack], ignore_index=True)
        results[index] = cam_out

    for case in CAMBIUM_CASES:
        gas_mult = 1
        if case in ("HighNGPrice", "LowRECost_HighNGPrice"):
            gas_mult = 1.385123
        if case in ("LowNGPrice", "HighRECost_LowNGPrice"):
            gas_mult = 0.731231

        ba_list_full = [f"p{n}" for n in range(1, 135)]
        ba_list_full.remove("p119")
        ba_list_full.remove("p122")
        ba_list_full.append("z122")

        threads = [None] * num_thread
        results = [None] * num_thread

        num_per_thread = int(round(len(ba_list_full) / num_thread, 0) + 1)
        ba_index = list(chunks(ba_list_full, num_per_thread))

        for i in range(len(threads)):
            threads[i] = Thread(target=run_region, args=(ba_index[i], results, i, case))
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()

        full_out = pd.DataFrame()
        for i in range(len(threads)):
            full_out = pd.concat([full_out, results[i]], ignore_index=True)

        full_out["r"] = full_out["r"].str.replace("z122", "p122")
        full_out_stack = full_out.copy()
        full_out_stack["case"] = case
        cambium_min = pd.concat([cambium_min, full_out_stack], ignore_index=True)

        h2_8760 = h2_char_orig.copy()
        h2_8760 = h2_8760[["pathway", "cost_cap", "cost_fom_per_metric_ton", "ccs_cap_rate_comb", "cost_vom", "int_elec", "int_ng"]]

        ba_to_state = c2z[["ba", "state_abb"]].drop_duplicates()
        ba_to_state.columns = ["r", "state_abb"]

        full_out_merged = full_out.merge(ba_to_state, how="left")
        full_out_merged = full_out_merged.merge(h2_8760, how="cross")
        full_out_merged["state_abb"] = to_lower_nospace(full_out_merged["state_abb"])
        full_out_merged = full_out_merged.merge(fuel_prices_state2, on="state_abb", how="left")
        full_out_merged["cf"] = full_out_merged["rank"] / 8760

        wacc_par3 = calc_wacc()
        crf_par3 = calc_crf(wacc_par3, FAC_LIFETIME)

        full_out_merged["cost_cap"] = full_out_merged["cost_cap"].astype(float)
        full_out_merged["LCOF_cap"] = full_out_merged["cost_cap"] / 114.877 / 365 * CDT_FACTOR * crf_par3 / full_out_merged["cf"]
        full_out_merged["LCOF_vom"] = full_out_merged["cost_vom"].astype(float)
        full_out_merged["emit_captured"] = full_out_merged["ccs_cap_rate_comb"].astype(float) * EMIT_NG * full_out_merged["int_ng"].astype(float)
        full_out_merged["LCOF_co2_tns"] = full_out_merged["ccs_cost"].astype(float) * full_out_merged["emit_captured"]
        full_out_merged["LCOF_energy_gas"] = gas_mult * full_out_merged["int_ng"].astype(float) * full_out_merged["gas_price"]
        full_out_merged["LCOF_energy_elec"] = full_out_merged["int_elec"].astype(float) * full_out_merged["avg_lcoe"]
        full_out_merged["LCOF_fom"] = full_out_merged["cost_fom_per_metric_ton"].astype(float) / 114.877

        h2_transport_stor2 = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "h2_transport_and_storage_costs.csv"),
                                         names=["type", "t", "parameter", "value"], header=0)
        h2_stor2 = pd.read_csv(os.path.join(lcoh_dir, "raw_data", "h2_storage_rb.csv"),
                               names=["type", "r"], header=0)
        h2_stor_char2 = pd.merge(h2_stor2, h2_transport_stor2, how="left")
        h2_stor_char2 = h2_stor_char2[h2_stor_char2["parameter"].isin(["cost_cap", "fom"])]
        h2_stor_char2["LCOS"] = 0
        h2_stor_char2.loc[h2_stor_char2["parameter"] == "cost_cap", "LCOS"] = (h2_stor_char2["value"] / 8760 / 114.877) / 0.15
        h2_stor_char2.loc[h2_stor_char2["parameter"] == "fom", "LCOS"] = h2_stor_char2["value"] / 1e4
        h2_stor_char2 = h2_stor_char2[h2_stor_char2["t"] == LCOE_YEAR]
        h2_stor_char2 = h2_stor_char2[["r", "parameter", "LCOS"]]
        h2_stor_char2 = h2_stor_char2.pivot(index="r", columns="parameter", values="LCOS").reset_index()
        h2_stor_char2.columns = ["r", "LCOS_cap", "LCOS_fom"]
        full_out_merged = full_out_merged.merge(h2_stor_char2, on="r", how="left")

        out_merged = full_out_merged[full_out_merged["pathway"].isin(TECH_KEEP)]
        col_keep = [
            "avg_lcoe", "rank", "r", "state_abb", "pathway", "cf", "LCOF_cap", "int_elec",
            "LCOF_vom", "emit_captured", "LCOF_co2_tns", "LCOF_energy_gas", "LCOF_energy_elec", "LCOF_fom",
            "LCOS_cap", "LCOS_fom"
        ]
        out_sub = out_merged[col_keep]
        ranks_keep = list(range(1, 8761, 4))
        out_sub = out_sub[out_sub["rank"].isin(ranks_keep)]

        retail_out = pd.DataFrame()
        for adder in RETAIL_ADDERS:
            temp = out_sub.copy()
            temp["retail_adder"] = adder
            retail_out = pd.concat([retail_out, temp], ignore_index=True)

        if case == "MidCase":
            retail_out.to_csv(os.path.join(lcoh_dir, "ba_retail_adders.csv"), index=False)

        retail_out_plot = retail_out[(retail_out["retail_adder"].isin([0, 30])) &
                                     (retail_out["pathway"].isin(["h2_pem_electrol_gh500", "h2_pem_electrol_gh1000", "h2_smr"]))]
        if case == "MidCase":
            retail_out_plot.to_csv(os.path.join(lcoh_dir, "retail_map.csv"), index=False)

        lcos_cam = retail_out.copy()
        lcos_cam["lcoh"] = (
            lcos_cam["LCOF_cap"] + lcos_cam["LCOF_vom"] + lcos_cam["LCOF_co2_tns"]
            + lcos_cam["LCOF_energy_gas"] + lcos_cam["LCOF_energy_elec"] + lcos_cam["LCOF_fom"]
            + lcos_cam["int_elec"].astype(float) * lcos_cam["retail_adder"].astype(float)
        )
        lcos_cam = lcos_cam[["r", "state_abb", "pathway", "retail_adder", "lcoh", "cf", "avg_lcoe"]]
        lcos_minrows = lcos_cam.groupby(["r", "state_abb", "pathway", "retail_adder"]).lcoh.idxmin()
        lcos_min = lcos_cam.loc[lcos_minrows]
        lcos_min_stack = lcos_min.copy()
        lcos_min_stack["case"] = case
        lcoh_min_full = pd.concat([lcoh_min_full, lcos_min_stack], ignore_index=True)

        lcos_min = lcos_min.merge(steel_char_copy, how="cross")
        lcos_prices = finito_prices.copy()
        lcos_prices.columns = lcos_prices.columns.str.replace("int", "price")
        lcos_min = lcos_min.merge(lcos_prices, how="left")

        lcos_min["lcos_coal"] = lcos_min["int_coal"].astype(float) * lcos_min["price_coal"].astype(float)
        lcos_min["lcos_met_coal"] = lcos_min["int_met_coal"].astype(float) * lcos_min["price_met_coal"].astype(float)
        lcos_min["lcos_coke"] = lcos_min["int_coke"].astype(float) * lcos_min["price_coke"].astype(float)
        lcos_min["lcos_ddfo"] = lcos_min["int_ddfo"].astype(float) * lcos_min["price_ddfo"].astype(float)
        lcos_min["lcos_ng"] = (1 if case not in ("HighNGPrice", "LowRECost_HighNGPrice") else 1.385123)
        if case in ("LowNGPrice", "HighRECost_LowNGPrice"):
            lcos_min["lcos_ng"] = 0.731231
        lcos_min["lcos_ng"] = lcos_min["lcos_ng"] * lcos_min["int_ng"].astype(float) * lcos_min["price_ng"].astype(float)

        lcos_min["lcos_elec"] = lcos_min["int_elec"].astype(float) * (lcos_min["avg_lcoe"] + lcos_min["retail_adder"])
        lcos_min["lcos_rfo"] = lcos_min["int_rfo"].astype(float) * lcos_min["price_rfo"].astype(float)
        lcos_min["lcos_h2_feed"] = 114 * lcos_min["int_h2_feed"].astype(float) * lcos_min["lcoh"].astype(float)

        lcos_min["lcos_m_scrap"] = lcos_min["int_m_scrap"].astype(float) * 325
        ore_transport.columns = ["r", "ore_transport"]
        lcos_min = lcos_min.merge(ore_transport, on="r", how="left")
        lcos_min["lcos_m_ore"] = lcos_min["int_m_ore"].astype(float) * (111.06 + lcos_min["ore_transport"])

        lcos_min["lcos_vom"] = lcos_min["cost_vom"]
        lcos_min["lcos_cap"] = lcos_min["cost_cap"].astype(float) * CDT_FACTOR * crf_par3 / (0.01 + lcos_min["cf"])
        lcos_min["lcos_fom"] = lcos_min["cost_fom"].astype(float) / CAP_FACTOR

        lcos_min = lcos_min.merge(fuel_prices_state2[["state_abb", "ccs_cost"]], how="left")
        lcos_min["lcos_ccs"] = lcos_min["ccs_cost"].astype(float) * lcos_min["ccs_cap_rate_comb"].astype(float) * (
            (lcos_min["int_coal"].astype(float) * 0.095
             + lcos_min["int_ng"].astype(float) * 0.053
             + lcos_min["int_coke"].astype(float) * 0.113
             + lcos_min["int_ddfo"].astype(float) * 0.074
             + lcos_min["int_rfo"].astype(float) * 0.074
             + lcos_min["int_met_coal"].astype(float) * 0.093)
            + lcos_min["co2_rate_proc"].astype(float)
        )

        steel_char_cols2 = [
            "r", "state_abb", "pathway", "type", "retail_adder", "lcos_coal", "lcos_met_coal", "lcos_coke",
            "lcos_ddfo", "lcos_ng", "lcos_elec", "lcos_elec", "lcos_rfo", "lcos_h2_feed", "lcos_m_scrap",
            "lcos_m_ore", "lcos_vom", "lcos_cap", "lcos_fom", "lcos_ccs"
        ]
        steel_out2 = pd.melt(lcos_min[steel_char_cols2],
                             id_vars=["r", "state_abb", "pathway", "type", "retail_adder"]).drop_duplicates()
        steel_out2["case"] = case
        steel_full = pd.concat([steel_full, steel_out2], ignore_index=True)

    # finalize cambium products
    ranks_keep = list(range(1, 8761, 4))
    ranks_keep.append(8760)

    cambium_min_out = cambium_min[cambium_min["rank"].isin(ranks_keep)]
    cambium_min_out.to_csv(os.path.join(lcoh_dir, "cambium_lcoe_scenarios_v2.csv"), index=False)
    steel_full = steel_full.drop_duplicates()
    steel_full.to_csv(os.path.join(lcoh_dir, "steel_scenarios.csv"), index=False)
    lcoh_min_full.to_csv(os.path.join(lcoh_dir, "lcoh_min_scenarios.csv"), index=False)

    # comparison artifacts
    cambium_min_out2 = cambium_min[cambium_min["rank"].isin(ranks_keep)].copy()
    cambium_min_out2["proxy"] = cambium_min_out2["avg_lcoe"] / cambium_min_out2["rank"]
    cambium_min_out2 = cambium_min_out2[cambium_min_out2["proxy"] > 0]
    cambium_min_placeholder = cambium_min_out2[cambium_min_out2["rank"] == 8760]
    cambium_min_out2 = cambium_min_out2[cambium_min_out2["rank"] > 8000]

    cam_min_out = cambium_min_out2[["r", "case", "proxy"]].groupby(["r", "case"]).min().reset_index()
    cam_merge_temp = cambium_min_out2[["r", "case", "proxy", "avg_lcoe"]]
    cam_merge_out = cam_min_out.merge(cam_merge_temp, how="left")
    cambium_min_placeholder = cambium_min_placeholder[cam_merge_out.columns]
    cam_merge_out = cam_merge_out.rename(columns={"r": "ba"})
    cambium_min_placeholder = cambium_min_placeholder.rename(columns={"r": "ba"})

    lcoe_merge = temp_wide[["fips", "state_abb", "offgrid_ele_price"]]

    comp_merge_scen = pd.DataFrame()
    for i in cam_merge_out["case"].unique():
        mult = 1
        if i in ("HighRECost", "HighRECost_LowNGPrice"):
            # adjusting costs upward by the average cost increase for wind and solar in ATB
            mult = 1.2693556
        if i in ("LowRECost", "LowRECost_HighNGPrice"):
            # adjusting costs downward by the simple average cost decrease for wind and solar in ATB
            mult = 0.8256311
        comp_temp = lcoe_merge.copy()
        comp_temp["offgrid_ele_price"] = comp_temp["offgrid_ele_price"] * mult
        comp_temp["case"] = i
        comp_merge_scen = pd.concat([comp_merge_scen, comp_temp], ignore_index=True)

    comp_merge_scen = comp_merge_scen.merge(c2z[["fips", "ba"]], how="left")
    comp_merge = comp_merge_scen.merge(cam_merge_out, how="left")
    comp_merge.to_csv(os.path.join(lcoh_dir, "comp_adder.csv"), index=False)

    cam_merge_state = cambium_min_placeholder.copy()
    cam_merge_state = cam_merge_state[cam_merge_state["case"] == "MidCase"]
    cam_merge_state = cam_merge_state.merge(c2z[["ba", "state_abb"]].drop_duplicates(), how="left")
    cam_merge_state["state_abb"] = cam_merge_state["state_abb"].str.lower()
    cam_merge_fuels = cam_merge_state.merge(fuel_prices_state2[["state_abb", "grid_ele_price"]], how="left")
    cam_merge_fuels["grid_ele_price"] = 10 * cam_merge_fuels["grid_ele_price"]
    cam_merge_fuels.to_csv(os.path.join(lcoh_dir, "retail_opt_comp.csv"), index=False)


if __name__ == "__main__":
    #parser.add_argument("--root_dir", default="/Users/max/Documents/GitHub", help="Repository root containing the LCOH folder")
    #args = parser.parse_args()
    main(root_dir)
#    main(args.root_dir)
