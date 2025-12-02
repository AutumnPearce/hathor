import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d

# ----------------------------------------------------------------------
# Parameters
binary_path = "../../dataset_examples/halo_3517_gas.bin"
npix = 512
boxsize_mpc = 20.0          # physical box size in Mpc
redshift = 0.5
lmin, lmax = 12, 18

# ----------------------------------------------------------------------
# Load or generate data
if os.path.exists(binary_path):
    from scipy.io import FortranFile
    def read_megatron_cutout(ff):
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
        for i, _ in enumerate(header):
            data = ff.read_reals("float64")
            if i == 0:
                all_data = np.zeros((len(data), len(header)))
            all_data[:, i] = data
        df = {col: all_data[:, i] for i, col in enumerate(header)}
        ne = (10.0**df["nH"]) * df["H_II"]
        ne += ((0.24*(10.0**df["nH"])/0.76)/4.0) * (df["He_II"] + 2.0*df["He_III"])
        df["ne"] = ne
        dx_vals = 10.0**df["dx"]
        level = np.rint(20.0 - np.log2(dx_vals / dx_vals.min())).astype(int)
        df["level"] = level
        return df

    with FortranFile(binary_path, 'r') as f:
        data = read_megatron_cutout(f)

    x = data["x"]
    y = data["y"]
    z = data["z"]
    dx = 10.0**data["dx"]
    ne = data["ne"]
    level = data["level"]
else:
    # Synthetic dataset
    N = 200000
    x = np.random.uniform(-boxsize_mpc/2, boxsize_mpc/2, N)
    y = np.random.uniform(-boxsize_mpc/2, boxsize_mpc/2, N)
    z = np.random.uniform(-boxsize_mpc/2, boxsize_mpc/2, N)
    # Approximate cell size: uniform log spacing within the box
    dx = np.full(N, np.log10(boxsize_mpc / np.cbrt(N)))
    dx = 10.0**dx
    ne = np.random.exponential(scale=1e-3, size=N)
    level = np.random.randint(lmin, lmax + 1, size=N)

# ----------------------------------------------------------------------
# Filter by level range
mask = (level >= lmin) & (level <= lmax)
x = x[mask]; y = y[mask]; z = z[mask]; dx = dx[mask]; ne = ne[mask]

# ----------------------------------------------------------------------
# Compute cell volume and weight for column density (ne * volume)
vol = dx**3
weight = ne * vol

# ----------------------------------------------------------------------
# Define projection box (centered on median, fixed physical size)
center = np.median(np.vstack([x, y, z]), axis=1)
half = boxsize_mpc / 2.0
xmin, xmax = center[0] - half, center[0] + half
ymin, ymax = center[1] - half, center[1] + half

# ----------------------------------------------------------------------
# Project along z using 2‑D histogram (sum of weights)
H, _, _, _ = binned_statistic_2d(
    x, y,
    values=weight,
    statistic='sum',
    bins=npix,
    range=[[xmin, xmax], [ymin, ymax]]
)

# Replace NaNs (empty pixels) with zero
image = np.nan_to_num(H, nan=0.0)

# ----------------------------------------------------------------------
# Plot
plt.figure(figsize=(6, 6))
plt.imshow(
    image.T,
    origin='lower',
    extent=[xmin, xmax, ymin, ymax],
    norm=LogNorm(
        vmin=image[image > 0].min() if np.any(image > 0) else 1e-10,
        vmax=image.max() if image.max() > 0 else 1
    ),
    cmap='inferno'
)
plt.colorbar(label=r'Electron column density $\Sigma_e$ (arb. units)')
plt.xlabel('x [Mpc]')
plt.ylabel('y [Mpc]')
plt.title(f'Halo 3517 Σ_e projection (z={redshift})')
plt.tight_layout()
plt.savefig('halo_3517_sigmae.png', dpi=300)
plt.close()