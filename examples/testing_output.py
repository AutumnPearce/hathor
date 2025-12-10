# ==============================
# cooling_vs_dyn.py
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import os


# -------------------------------------------------
# 1. Helper: read megaton gas cutout (provided)
# -------------------------------------------------
def read_megatron_cutout(filepath, lmax=20.0, boxsize=50.0, h=0.672699966430664):
   """
   Reads a megatron gas cutout written as Fortran unformatted binary.
   Returns a pandas DataFrame with physical quantities.
   """
   header = [
       "redshift","dx","x","y","z","vx","vy","vz","nH","T","P",
       "nFe","nO","nN","nMg","nNe","nSi","nCa","nC","nS","nCO",
       "O_I","O_II","O_III","O_IV","O_V","O_VI","O_VII","O_VIII",
       "N_I","N_II","N_III","N_IV","N_V","N_VI","N_VII",
       "C_I","C_II","C_III","C_IV","C_V","C_VI",
       "Mg_I","Mg_II","Mg_III","Mg_IV","Mg_V","Mg_VI","Mg_VII","Mg_VIII","Mg_IX","Mg_X",
       "Si_I","Si_II","Si_III","Si_IV","Si_V","Si_VI","Si_VII","Si_VIII","Si_IX","Si_X","Si_XI",
       "S_I","S_II","S_III","S_IV","S_V","S_VI","S_VII","S_VIII","S_IX","S_X","S_XI",
       "Fe_I","Fe_II","Fe_III","Fe_IV","Fe_V","Fe_VI","Fe_VII","Fe_VIII","Fe_IX","Fe_X","Fe_XI",
       "Ne_I","Ne_II","Ne_III","Ne_IV","Ne_V","Ne_VI","Ne_VII","Ne_VIII","Ne_IX","Ne_X",
       "H_I","H_II","He_II","He_III",
       "Habing","Lyman_Werner","HI_Ionising","H2_Ionising","HeI_Ionising","HeII_ionising"
   ]


   with FortranFile(filepath, 'r') as ff:
       # read everything column by column
       ncols = len(header)
       # read first column to know how many rows we have
       first = ff.read_reals("float64")
       nrows = len(first)
       data = np.empty((nrows, ncols))
       data[:, 0] = first
       for i in range(1, ncols):
           data[:, i] = ff.read_reals("float64")


   df = pd.DataFrame(data, columns=header)


   # ---- compute derived quantities ----
   # cell volume (cm^3)
   cell_dx_cm = 10.0 ** df["dx"]          # cm (dx is log10(cm))
   cell_vol = cell_dx_cm ** 3


   # mass density: ρ = nH * μ * m_p ; μ≈0.59 for fully ionized gas
   m_p = 1.6726219e-24   # g
   mu = 0.59
   nH = 10.0 ** df["nH"]          # cm^-3
   rho = nH * mu * m_p            # g cm^-3
   df["mass"] = rho * cell_vol    # g


   # total particle density (includes He). Approx n = 1.92 nH for primordial fully ionized gas
   df["n_tot"] = 1.92 * nH


   # temperature in K
   df["T_K"] = 10.0 ** df["T"]


   # positions in cm (the simulation stores coordinates in units of the box length / (1+z))
   # Convert to physical kpc: 1 cm = 3.24078e-22 kpc
   cm_to_kpc = 3.24078e-22
   # Option 1: Positions are already in kpc
   df["x_kpc"] = df["x"]
   df["y_kpc"] = df["y"]
   df["z_kpc"] = df["z"]


   return df


def read_megatron_star_cutout(filepath):
   # Not needed for the current analysis – placeholder
   return pd.DataFrame()


# -------------------------------------------------
# 2. Load data
# -------------------------------------------------
data_dir = "/Users/autumn/Documents/GitHub/hathor/test_data"
gas_path = os.path.join(data_dir, "halo_1411_gas.bin")
star_path = os.path.join(data_dir, "halo_1411_stars.bin")


print("Reading gas cutout ...")
gas = read_megatron_cutout(gas_path)


print("Reading star cutout (unused) ...")
stars = read_megatron_star_cutout(star_path)


# -------------------------------------------------
# 3. Geometry: centre & radius
# -------------------------------------------------
# mass‑weighted centre (kpc)
center = np.array([
   np.average(gas["x_kpc"], weights=gas["mass"]),
   np.average(gas["y_kpc"], weights=gas["mass"]),
   np.average(gas["z_kpc"], weights=gas["mass"])
])


# radial distance of each cell from centre (kpc)
dx = gas["x_kpc"] - center[0]
dy = gas["y_kpc"] - center[1]
dz = gas["z_kpc"] - center[2]
r_kpc = np.sqrt(dx**2 + dy**2 + dz**2)
gas["r_kpc"] = r_kpc


# Estimate virial radius – 95th percentile
Rvir = np.percentile(r_kpc, 95)
print(f"Estimated Rvir = {Rvir:.2f} kpc")


# -------------------------------------------------
# 4. Radial bins (linear in r/Rvir)
# -------------------------------------------------
nbins = 50
r_edges = np.linspace(0.0, Rvir, nbins + 1)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
r_norm = r_centers / Rvir


# -------------------------------------------------
# 5. Enclosed mass profile
# -------------------------------------------------
# cumulative mass as a function of radius
sorted_idx = np.argsort(gas["r_kpc"])
sorted_r = gas["r_kpc"].values[sorted_idx]
sorted_m = gas["mass"].values[sorted_idx]
cum_mass = np.cumsum(sorted_m)   # g


# Interpolate M(<r) at the bin centres (in kpc -> convert to cm)
kpc_to_cm = 3.085677581e21
r_cm = r_centers * kpc_to_cm
M_enc = np.interp(r_centers, sorted_r, cum_mass)   # g


# -------------------------------------------------
# 6. Dynamical time
# -------------------------------------------------
G_cgs = 6.67430e-8   # cm^3 g^-1 s^-2
r_cgs = r_cm
V_circ = np.sqrt(G_cgs * M_enc / r_cgs)          # cm s^-1
t_dyn = r_cgs / V_circ                           # s


# -------------------------------------------------
# 7. Cooling function Λ(T, Z)
# -------------------------------------------------
k_B = 1.380649e-16    # erg K^-1
Lambda0 = 1e-22       # erg cm^3 s^-1 (normalisation)


def cooling_rate(T, Zsun_ratio):
   """
   Simple analytic cooling function.
   T : temperature in K (array)
   Zsun_ratio : metallicity relative to solar (scalar)
   Returns Λ in erg cm^3 s^-1
   """
   # Power‑law temperature dependence (≈1/T) and linear Z dependence
   return Lambda0 * (T / 1e6)**-1.0 * (0.1 + 0.9 * Zsun_ratio)


# Two metallicities for uncertainty band
Z_low = 0.1   # 0.1 Zsun
Z_high = 1.0  # 1.0 Zsun


# -------------------------------------------------
# 8. Cooling time per cell (cgs)
# -------------------------------------------------
n_tot = gas["n_tot"].values            # cm^-3
T_cell = gas["T_K"].values              # K


Lambda_low  = cooling_rate(T_cell, Z_low)
Lambda_high = cooling_rate(T_cell, Z_high)


t_cool_low  = (1.5 * n_tot * k_B * T_cell) / Lambda_low   # s
t_cool_high = (1.5 * n_tot * k_B * T_cell) / Lambda_high  # s


gas["t_cool_low"]  = t_cool_low
gas["t_cool_high"] = t_cool_high


# -------------------------------------------------
# 9. Ratio t_cool / t_dyn in the radial range 0.5–1.0 Rvir
# -------------------------------------------------
mask_shell = (gas["r_kpc"] >= 0.5 * Rvir) & (gas["r_kpc"] <= Rvir)


# Bin the ratios (median and percentiles) separately for the two metallicities
def radial_stats(r, values, bins):
   """Return median, 16th, 84th percentiles per radial bin."""
   inds = np.digitize(r, bins) - 1
   med, p16, p84 = [], [], []
   for i in range(len(bins)-1):
       bin_vals = values[inds == i]
       if len(bin_vals) == 0:
           med.append(np.nan)
           p16.append(np.nan)
           p84.append(np.nan)
       else:
           med.append(np.nanmedian(bin_vals))
           p16.append(np.nanpercentile(bin_vals, 16))
           p84.append(np.nanpercentile(bin_vals, 84))
   return np.array(med), np.array(p16), np.array(p84)


# -------------------------------------------------
# 9. Ratio t_cool / t_dyn in the radial range 0.5–1.0 Rvir
# -------------------------------------------------
mask_shell = (gas["r_kpc"] >= 0.5 * Rvir) & (gas["r_kpc"] <= Rvir)


# Interpolate t_dyn to each cell's radius
t_dyn_at_cells = np.interp(gas["r_kpc"], r_centers, t_dyn)


# Compute ratio for each cell
gas["ratio_low"] = gas["t_cool_low"] / t_dyn_at_cells
gas["ratio_high"] = gas["t_cool_high"] / t_dyn_at_cells


# Now extract the ratios for the shell region
ratio_low = gas.loc[mask_shell, "ratio_low"].values
ratio_high = gas.loc[mask_shell, "ratio_high"].values
r_shell = gas.loc[mask_shell, "r_kpc"].values


# Bin the ratios (median and percentiles) separately for the two metallicities
def radial_stats(r, values, bins):
   """Return median, 16th, 84th percentiles per radial bin."""
   inds = np.digitize(r, bins) - 1
   med, p16, p84 = [], [], []
   for i in range(len(bins)-1):
       bin_vals = values[inds == i]
       if len(bin_vals) == 0:
           med.append(np.nan)
           p16.append(np.nan)
           p84.append(np.nan)
       else:
           med.append(np.nanmedian(bin_vals))
           p16.append(np.nanpercentile(bin_vals, 16))
           p84.append(np.nanpercentile(bin_vals, 84))
   return np.array(med), np.array(p16), np.array(p84)


med_low, p16_low, p84_low   = radial_stats(r_shell, ratio_low, r_edges)
med_high, p16_high, p84_high = radial_stats(r_shell, ratio_high, r_edges)


# For plotting we use the median of the low‑Z case as the central line,
# and the envelope given by the two metallicities.
med_ratio = med_low
lower_band = np.minimum(p16_low, p16_high)
upper_band = np.maximum(p84_low, p84_high)


med_low, p16_low, p84_low   = radial_stats(r_shell, ratio_low, r_edges)
med_high, p16_high, p84_high = radial_stats(r_shell, ratio_high, r_edges)


# For plotting we use the median of the low‑Z case as the central line,
# and the envelope given by the two metallicities.
med_ratio = med_low
lower_band = np.minimum(p16_low, p16_high)
upper_band = np.maximum(p84_low, p84_high)


# -------------------------------------------------
# 10. Plot
# -------------------------------------------------
plt.figure(figsize=(6,4))
plt.fill_between(r_norm, lower_band, upper_band, color="C1", alpha=0.3,
                label=r"$Z=0.1-1\,Z_\odot$ uncertainty")
plt.plot(r_norm, med_ratio, color="C1", lw=2, label=r"$t_{\rm cool}/t_{\rm dyn}$ (median)")


# Horizontal line at unity with 1σ (10 %) band
plt.axhline(1.0, color="k", ls="--")
plt.fill_between(r_norm, 0.9, 1.1, color="gray", alpha=0.2, label="±10 %")


plt.xlim(0.5, 1.0)
plt.yscale("log")
plt.xlabel(r"$r/R_{\rm vir}$")
plt.ylabel(r"$t_{\rm cool}/t_{\rm dyn}$")
plt.title("Cooling vs Dynamical time in the CGM")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
