import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import FortranFile
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.io import FortranFile
import cutout_utils as cu
from cutout_utils import read_megatron_cutout

# --- If the functions are in a file like "megatron_tools.py", import them:
# from megatron_tools import read_megatron_cutout, make_image
# Otherwise, if they’re defined in your notebook already, you can skip the import.

# === 1️⃣ Read the binary file ===
fname = "../../dataset_examples/halo_3517_gas.bin"
ff = FortranFile(fname, 'r')
gas_df = read_megatron_cutout(ff)
ff.close()

print("✅ File loaded.")
print("Columns:", gas_df.columns[:10], "...")  # show first few columns
print("Number of cells:", len(gas_df))

# === 2️⃣ Check shapes and data types ===
print("\n--- Data sanity check ---")
print("positions shape:", gas_df[["x", "y", "z"]].values.shape)
print("level shape:", gas_df["level"].shape)
print("ne shape:", gas_df["ne"].shape)
print("dx shape:", gas_df["dx"].shape)
print("\nDtypes:\n", gas_df[["x","y","z","level","ne","dx"]].dtypes)

# === 3️⃣ Convert all arrays to numeric numpy arrays ===
positions = gas_df[["x","y","z"]].values.astype(float)
levels = gas_df["level"].values.astype(int)
features = gas_df["ne"].values.astype(float)
dx = gas_df["dx"].values.astype(float)

# Optional: drop NaNs if present
mask = np.isfinite(features) & np.isfinite(levels) & np.isfinite(dx)
positions = positions[mask]
levels = levels[mask]
features = features[mask]
dx = dx[mask]


print(f"Cleaned data: {len(features)} valid cells remain.")

# === 4️⃣ Create 2D image ===
image, _ = cu.make_image(
    positions,
    levels,
    features,
    dx,
    view_dir="z",
    npix=512,
    lmin=12,
    lmax=18,
    redshift=0.5,
    boxsize=20.0
)

# === 5️⃣ Plot result ===
plt.figure(figsize=(6,6))
plt.imshow(image, norm=LogNorm(), cmap="magma", origin="lower")
# plt.imshow(image, cmap="magma", origin="lower")
plt.colorbar(label="Electron density (arbitrary units)")
plt.xlabel("x [kpc]")
plt.ylabel("y [kpc]")
plt.title("Projected Electron Density (halo_3517)")
plt.show()
# plt.savefig("projected_electron_density.png", dpi=300)
