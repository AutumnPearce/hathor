# build_rascas_domain_and_ics.py
import json
import os
import numpy as np
import pandas as pd
from scipy.io import FortranFile

# ---- If you have your own helper, import it; else fall back to the local reader ----
try:
    import cutout_utils as cu
    READ_FROM_CUT = cu.read_megatron_cutout
except Exception:
    READ_FROM_CUT = None

# -----------------------------
# Physical constants / helpers
# -----------------------------
clight  = 2.99792458e10          # [cm/s]
kb      = 1.38064852e-16         # [erg/K]
mp      = 1.672621898e-24        # [g]
planck  = 6.626070040e-27        # [erg s]
avogad  = 6.0221409e23
X_H     = 0.76
element_weights = { 'H': 1.00797 }
for k in element_weights:
    element_weights[k] /= avogad

# Lyα: 1215.67 Angstrom
lambda_lya_cm = 1215.67e-8          # [cm]
nu0_lya = clight / lambda_lya_cm    # [Hz]
E_lya   = planck * nu0_lya          # [erg]

def isotropic_direction_fast(n):
    phi   = 2.0 * np.pi * np.random.random(n)
    cos_t = 1.0 - 2.0 * np.random.random(n)  # in [-1,1]
    sin_t = np.sqrt(1.0 - cos_t**2)
    k = np.zeros((n,3))
    k[:,0] = sin_t * np.cos(phi)
    k[:,1] = sin_t * np.sin(phi)
    k[:,2] = cos_t
    k /= np.linalg.norm(k, axis=1)[:,None]
    return k

# -----------------------------
# 1) Read cutout
# -----------------------------
def read_megatron_cutout_local(ff, lmax=20.0):
    """Minimal inline reader in case cutout_utils is unavailable."""
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
    for i,_ in enumerate(header):
        dat = ff.read_reals("float64")
        if i == 0:
            all_dat = np.zeros((len(dat), len(header)))
        all_dat[:,i] = dat
    df = pd.DataFrame(all_dat, columns=header)

    # Electron density estimate
    edens = (10.**df["nH"]) * df["H_II"]  # H contribution
    edens += ((0.24*(10.**df["nH"])/0.76)/4.0) * (df["He_II"] + 2.0*df["He_III"])  # He
    df["ne"] = edens

    # AMR level (if useful downstream)
    level = np.rint(20.0 - np.log2((10.0**df["dx"])/(10.0**df["dx"].min()))).astype(int)
    df["level"] = level
    return df

def read_cutout(path):
    with FortranFile(path, "r") as ff:
        if READ_FROM_CUT is not None:
            return READ_FROM_CUT(ff)
        return read_megatron_cutout_local(ff)

# -----------------------------
# 2) Build uniform domain (CIC)
# -----------------------------
def cic_deposit(pos, boxlen, Ngrid, qty, mass_weight=None):
    """
    CIC deposition on a uniform grid in [0, boxlen]^3 with positions in same units.
    qty: values per particle/cell (e.g., density * volume for mass deposition).
    mass_weight: optional weights for mass-weighted fields (e.g., mass); if None, uses qty as weights directly.
    Returns array shape (Ngrid,Ngrid,Ngrid), x-major storage (x fastest).
    """
    grid = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    if mass_weight is None:
        w = qty
    else:
        w = mass_weight

    # Grid spacing
    dx = boxlen / Ngrid

    # Normalize positions into [0, boxlen)
    r = np.mod(pos, boxlen)
    gx = r[:,0]/dx
    gy = r[:,1]/dx
    gz = r[:,2]/dx

    i = np.floor(gx).astype(int) % Ngrid
    j = np.floor(gy).astype(int) % Ngrid
    k = np.floor(gz).astype(int) % Ngrid

    tx = gx - i
    ty = gy - j
    tz = gz - k

    # Neighbors with periodic wrap
    ip = (i + 1) % Ngrid
    jp = (j + 1) % Ngrid
    kp = (k + 1) % Ngrid

    wx0 = 1.0 - tx; wx1 = tx
    wy0 = 1.0 - ty; wy1 = ty
    wz0 = 1.0 - tz; wz1 = tz

    # 8 corners
    np.add.at(grid, (i , j , k ), w * wx0 * wy0 * wz0)
    np.add.at(grid, (ip, j , k ), w * wx1 * wy0 * wz0)
    np.add.at(grid, (i , jp, k ), w * wx0 * wy1 * wz0)
    np.add.at(grid, (i , j , kp), w * wx0 * wy0 * wz1)
    np.add.at(grid, (ip, jp, k ), w * wx1 * wy1 * wz0)
    np.add.at(grid, (ip, j , kp), w * wx1 * wy0 * wz1)
    np.add.at(grid, (i , jp, kp), w * wx0 * wy1 * wz1)
    np.add.at(grid, (ip, jp, kp), w * wx1 * wy1 * wz1)

    return grid

def build_uniform_domain_from_cut(df, Ngrid=256):
    """
    Voxelize the cutout onto a uniform grid using CIC.
    Assumes positions x,y,z in *box units* in [0, Lbox). If your cutout isn't already normalized,
    we remap to [xmin,xmax] etc. and take boxlen = max_range for cubic box.
    """
    # --- Box geometry in "box units" as given by the cutout ---
    x = df["x"].to_numpy(); y = df["y"].to_numpy(); z = df["z"].to_numpy()
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    # Make it cubic (use the largest span)
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    Lbox = float(max(Lx, Ly, Lz))
    # Shift to [0, Lbox)
    pos = np.column_stack([x - xmin, y - ymin, z - zmin])

    # Physical fields per cell center
    nH   = 10.0**df["nH"].to_numpy()          # [cm^-3]
    T    = 10.0**df["T"].to_numpy()           # [K]
    ne   = df["ne"].to_numpy()                # [cm^-3]
    xHI  = df["H_I"].to_numpy()
    xHII = df["H_II"].to_numpy()
    nHI  = nH * xHI
    nHII = nH * xHII

    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()
    vz = df["vz"].to_numpy()
    # If velocities are in code units, leave as-is; RASCAS can be given any consistent units as long as you document them.

    # Mass per cell (approx): rho * V  with  rho ~ nH * m_H / X_H
    dx_cell = 10.0**df["dx"].to_numpy()      # in same box units as x,y,z
    V_cell  = dx_cell**3
    rho     = nH * element_weights["H"] / X_H    # [g cm^-3] if nH in cm^-3 and units compatible
    # We don't know absolute cm↔box conversion here, so keep weights dimensionless for deposition purposes.
    # Use "mass-like" weight as nH * V_boxunits to mass-weight other fields consistently on the grid:
    w_mass = nH * V_cell

    # --- Deposit "mass" (w_mass) to get a weight grid ---
    W = cic_deposit(pos, Lbox, Ngrid, qty=w_mass)

    # Helper to do mass-weighted deposition of a field 'f'
    def mass_weighted_grid(f):
        G = cic_deposit(pos, Lbox, Ngrid, qty=f*w_mass)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(W>0, G/W, 0.0)
        return out

    # Volume-weighted helper (replace weight with cell volume if you prefer)
    V = cic_deposit(pos, Lbox, Ngrid, qty=V_cell)
    def volume_weighted_grid(f):
        G = cic_deposit(pos, Lbox, Ngrid, qty=f*V_cell)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(V>0, G/V, 0.0)
        return out

    # --- Build grids ---
    nH_grid   = volume_weighted_grid(nH)      # number density typically volume-weighted
    nHI_grid  = volume_weighted_grid(nHI)
    nHII_grid = volume_weighted_grid(nHII)
    ne_grid   = volume_weighted_grid(ne)
    T_grid    = mass_weighted_grid(T)         # temperature often mass-weighted
    vx_grid   = mass_weighted_grid(vx)
    vy_grid   = mass_weighted_grid(vy)
    vz_grid   = mass_weighted_grid(vz)

    domain = {
        "nH": nH_grid, "nHI": nHI_grid, "nHII": nHII_grid, "ne": ne_grid,
        "T": T_grid, "vx": vx_grid, "vy": vy_grid, "vz": vz_grid,
        "W_masslike": W, "V_cell": V,
        "boxlen": Lbox, "Ngrid": int(Ngrid),
        "origin_shift": [float(xmin), float(ymin), float(zmin)],
        "units": {
            "nH":"cm^-3","nHI":"cm^-3","nHII":"cm^-3","ne":"cm^-3",
            "T":"K",
            "v":"(code units – same as input)",
            "boxlen":"box units (same as input)"
        },
        "order":"C (x,y,z) with x fastest"
    }
    return domain

def save_domain_npz(domain, out_npz="DATA/rascas_domain_uniform.npz", out_json="DATA/rascas_domain_uniform.json"):
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz,
        nH=domain["nH"], nHI=domain["nHI"], nHII=domain["nHII"], ne=domain["ne"],
        T=domain["T"], vx=domain["vx"], vy=domain["vy"], vz=domain["vz"],
        W_masslike=domain["W_masslike"], V_cell=domain["V_cell"]
    )
    meta = {
        "boxlen": domain["boxlen"],
        "Ngrid": domain["Ngrid"],
        "origin_shift": domain["origin_shift"],
        "units": domain["units"],
        "order": domain["order"],
        "fields": ["nH","nHI","nHII","ne","T","vx","vy","vz"],
        "npz_path": out_npz
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    return out_npz, out_json

# -----------------------------
# 3) Lyman-alpha ICs
# -----------------------------
def compute_lya_luminosity_per_cell(df):
    """
    Recombination + collisional excitation; same formulae you used.
    """
    nH   = 10.0**df["nH"].to_numpy()
    T    = 10.0**df["T"].to_numpy()
    ne   = df["ne"].to_numpy()
    xHII = df["H_II"].to_numpy()
    xHI  = df["H_I"].to_numpy()
    nHII = nH * xHII
    nHI  = nH * xHI

    # Case B recombination coefficient (Hui & Gnedin 1997)
    alphaB = 2.59e-13 * (T / 1e4)**-0.7   # [cm^3 s^-1]

    # Collisional excitation emissivity (Katz+ 2022 style)
    a, b, c, d = 6.58e-18, 4.86e4, 0.185, 0.895
    epsilon_coll = a / (T**c) * np.exp(-b / (T**d))

    dx = 10.0**df["dx"].to_numpy()   # box units
    emis_rec  = ne * nHII * alphaB * E_lya                  # [erg cm^-3 s^-1]
    emis_coll = ne * nHI  * epsilon_coll                    # [erg cm^-3 s^-1]
    L_Lya     = (emis_rec + emis_coll) * (dx**3)            # [erg s^-1]   (up to box↔cm factor)
    return L_Lya

def write_ics(
    N_phot, L_Lya, df, out_path="DATA/test_rascas_Lya_ics.bin", seed=42
):
    """
    Creates ICs in the Fortran unformatted format:
    [int32 N_phot] [float64 N_real_phot] [int32 seed]
    [int64 ids(N)] [float64 nu_ext(N)] [float64 pos(N,3)] [float64 dirs(N,3)]
    [int32 seeds(N)] [float64 vel(N,3)]
    """
    rng = np.random.default_rng(seed)

    # Allocate photons to cells by luminosity
    probs = L_Lya / L_Lya.sum()
    cell_phot = rng.multinomial(N_phot, probs)

    # Cell centers & sizes
    centers = df[["x","y","z"]].to_numpy()
    vcell   = df[["vx","vy","vz"]].to_numpy()
    dx_cell = (10.0**df["dx"].to_numpy()).reshape(-1,1)

    # Repeat rows per cell_phot
    pos_centers = np.repeat(centers, cell_phot, axis=0)
    vel_source  = np.repeat(vcell,   cell_phot, axis=0)
    dx_rep      = np.repeat(dx_cell, cell_phot, axis=0)

    # Uniformly sample within cube of side dx
    offsets = (rng.random(pos_centers.shape) - 0.5) * dx_rep
    pos = pos_centers + offsets

    # Isotropic directions
    dirs = isotropic_direction_fast(N_phot)

    # Thermal Doppler sigma in frequency space:
    # v_th = sqrt(2 k T / m_H); sigma_nu = nu0 * (v_th / c)
    T = 10.0**df["T"].to_numpy()
    vth = np.sqrt(2.0 * kb * T / element_weights["H"])      # [cm/s]
    sigma_nu_cell = nu0_lya * (vth / clight)                # [Hz]
    sigma_rep = np.repeat(sigma_nu_cell, cell_phot)

    # Frequencies in cell frame (Gaussian around nu0)
    nu_cell = rng.normal(loc=nu0_lya, scale=sigma_rep)

    # Bulk velocity shift: nu_ext = nu_cell / (1 - (v·k)/c)
    scal = (vel_source * dirs).sum(axis=1)
    nu_ext = nu_cell / (1.0 - (scal / clight))

    # Seeds per photon (optional; some codes ignore)
    seeds = np.arange(N_phot, dtype=np.int32) + 1
    ids   = np.arange(N_phot, dtype=np.int64) + 1

    # Total *real* photons per second represented by these N_phot packets
    N_real_phot = float(L_Lya.sum() / E_lya)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FortranFile(out_path, 'w') as f:
        f.write_record(np.int32(N_phot))
        f.write_record(np.float64(N_real_phot))
        f.write_record(np.int32(seed))
        f.write_record(ids.astype(np.int64))
        f.write_record(nu_ext.astype(np.float64))
        f.write_record(pos.astype(np.float64))
        f.write_record(dirs.astype(np.float64))
        f.write_record(seeds.astype(np.int32))
        f.write_record(vel_source.astype(np.float64))
    return out_path, N_real_phot

# -----------------------------
# Main example
# -----------------------------
if __name__ == "__main__":
    CUT_PATH = "DATA/halo_3517_gas.bin"
    DOMAIN_NGRID = 256          # change to taste (128/256/384/512)
    N_PHOT = 1_000_000

    print("📦 Reading cutout...")
    df_gas = read_cutout(CUT_PATH)

    print("🧱 Building uniform domain (CIC)...")
    domain = build_uniform_domain_from_cut(df_gas, Ngrid=DOMAIN_NGRID)
    npz_path, json_path = save_domain_npz(domain)
    print(f"✅ Domain saved:\n  {npz_path}\n  {json_path}")

    print("💡 Computing Lyα luminosity per cell...")
    L_Lya = compute_lya_luminosity_per_cell(df_gas)

    print("🔦 Writing Lyα ICs...")
    ics_path, Nreal = write_ics(N_PHOT, L_Lya, df_gas, out_path="DATA/test_rascas_Lya_ics.bin", seed=42)
    print(f"✅ ICs saved: {ics_path}")
    print(f"   Real photon rate represented: {Nreal:.3e} s^-1")
