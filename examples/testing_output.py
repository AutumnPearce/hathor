
"""
Projected metallicity map of the CGM with recent star particles overlay.

Steps:
1. Load the binary cutouts (gas and stars) using the helper functions
   `read_megatron_cutout` and `read_megatron_star_cutout` (adapted from the
   example you provided).
2. Derive a simple metallicity proxy for each gas cell by summing the
   number‑densities of the most common metal species (C, N, O, Ne, Mg, Si,
   S, Fe).  The exact definition is not crucial for a visual inspection –
   the map is meant to highlight relative metal‑rich regions.
3. Compute the AMR level for each cell from its `dx` value – this is required
   by the `make_image` routine which builds a 2‑D projection by histogramming
   the cells on the finest level.
4. Produce a projected metallicity image (along the z‑axis) with the
   `make_image` function.  The image is returned in physical kpc units.
5. Load the star catalogue, select stars younger than 50 Myr, and compute a
   recent specific star‑formation rate proxy (`SFR ≈ mass / age`).  The stars
   are projected onto the same plane.
6. Estimate the halo centre as the median of the gas positions and define the
   virial radius `Rvir` as the 95‑th percentile of the 3‑D distances of gas
   cells from that centre.  This is a reasonable approximation when only a
   single snapshot is available.
7. Plot the metallicity map with a logarithmic colour scale, add a colour bar,
   draw metallicity contours at the 25‑, 50‑ and 75‑percentile levels, overlay
   the young stars (colour‑coded by the SFR proxy), draw a circle at `Rvir`
   and a set of concentric annuli (0.2 Rvir, 0.4 Rvir … 1.0 Rvir).

The script is designed to run on a typical laptop – the image resolution is
set to 1024×1024 pixels (adjustable via `npix`).  All heavy‑lifting is done
with NumPy histograms, which are fast and memory‑efficient.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.io import FortranFile
import os

# ---------------------------------------------------------------
# Helper functions – copy from the example you gave (slightly trimmed)
# ---------------------------------------------------------------

def read_megatron_cutout(filepath, lmax=20.0, boxsize=50.0, h=0.672699966430664):
    """Read a Megatron gas cutout and return a pandas DataFrame.
    The function reproduces the behaviour of the original example, but it
    does not compute every derived column – only those needed for this plot.
    """
    with FortranFile(filepath, 'r') as ff:
        header = [
            "redshift", "dx", "x", "y", "z",
            "vx", "vy", "vz", "nH", "T", "P",
            "nFe", "nO", "nN", "nMg", "nNe", "nSi", "nCa", "nC", "nS", "nCO",
            "O_I", "O_II", "O_III", "O_IV", "O_V", "O_VI", "O_VII", "O_VIII",
            "N_I", "N_II", "N_III", "N_IV", "N_V", "N_VI", "N_VII",
            "C_I", "C_II", "C_III", "C_IV", "C_V", "C_VI",
            "Mg_I", "Mg_II", "Mg_III", "Mg_IV", "Mg_V", "Mg_VI", "Mg_VII", "Mg_VIII", "Mg_IX", "Mg_X",
            "Si_I", "Si_II", "Si_III", "Si_IV", "Si_V", "Si_VI", "Si_VII", "Si_VIII", "Si_IX", "Si_X", "Si_XI",
            "S_I", "S_II", "S_III", "S_IV", "S_V", "S_VI", "S_VII", "S_VIII", "S_IX", "S_X", "S_XI",
            "Fe_I", "Fe_II", "Fe_III", "Fe_IV", "Fe_V", "Fe_VI", "Fe_VII", "Fe_VIII", "Fe_IX", "Fe_X", "Fe_XI",
            "Ne_I", "Ne_II", "Ne_III", "Ne_IV", "Ne_V", "Ne_VI", "Ne_VII", "Ne_VIII", "Ne_IX", "Ne_X",
            "H_I", "H_II", "He_II", "He_III",
            "Habing", "Lyman_Werner", "HI_Ionising", "H2_Ionising", "HeI_Ionising", "HeII_ionising"
        ]
        # Read all columns (they are stored one after another as double precision)
        all_data = []
        for _ in header:
            col = ff.read_reals('float64')
            all_data.append(col)
        all_arr = np.column_stack(all_data)
        df = pd.DataFrame(all_arr, columns=header)
    return df


def read_megatron_star_cutout(filepath):
    """Read a Megatron star cutout and return a pandas DataFrame."""
    with FortranFile(filepath, 'r') as ff:
        nstars = ff.read_ints('int32')[0]
        header = [
            "x", "y", "z",
            "vx", "vy", "vz",
            "age",
            "met_Fe", "met_O", "met_N", "met_Mg", "met_Ne", "met_Si", "met_Ca", "met_C", "met_S",
            "initial_mass", "mass",
        ]
        if nstars == 0:
            return pd.DataFrame(columns=header)
        all_data = []
        for _ in header:
            col = ff.read_reals('float64')
            all_data.append(col)
        all_arr = np.column_stack(all_data)
        df = pd.DataFrame(all_arr, columns=header)
    return df

# ---------------------------------------------------------------
# Image creation utilities (identical to the ones you supplied)
# ---------------------------------------------------------------

def make_image(positions, levels, feature, dx, lmin=13, lmax=18, npix=1024,
               redshift=6.0, boxsize=20.0, view_dir='z'):
    """Project a 3‑D field onto a 2‑D image using AMR levels.
    Returns the image and a cell‑count image (useful for normalisation).
    """
    # Physical size of the simulation box at the given redshift (pkpc)
    physical_boxsize = boxsize * 1000.0 / (1.0 + redshift)
    width = (npix / (2 ** lmax)) * physical_boxsize
    # Convert positions to pixel coordinates on the finest level
    pixel_positions = positions * (2 ** lmax)

    # Determine image centre (median of positions)
    cen = np.median(pixel_positions, axis=0)
    cen = np.rint(cen).astype(int)
    # Define the range for each axis on the finest level
    half = npix // 2
    ranges = []
    for i in range(3):
        low = cen[i] - half
        high = cen[i] + half
        ranges.append((low, high))

    # Choose which two axes to project onto
    if view_dir == 'x':
        im_range = (ranges[1], ranges[2])
        i1, i2 = 1, 2
    elif view_dir == 'y':
        im_range = (ranges[0], ranges[2])
        i1, i2 = 0, 2
    else:  # 'z'
        im_range = (ranges[0], ranges[1])
        i1, i2 = 0, 1

    # Number of pixels per level (coarse to fine)
    l_pix = [int(npix / (2 ** (lmax - l))) for l in range(lmin, lmax + 1)]

    image = np.zeros((npix, npix))
    image_cd = np.zeros((npix, npix))

    for lvl, pix in zip(range(lmin, lmax + 1), l_pix):
        if pix < 1:
            continue
        mask = levels == lvl
        if not np.any(mask):
            continue
        # 2‑D histogram weighted by the field * cell volume
        H, _, _ = np.histogram2d(
            pixel_positions[mask, i1],
            pixel_positions[mask, i2],
            bins=pix,
            range=im_range,
            weights=feature[mask] * (10.0 ** dx[mask]),
        )
        # Cell‑count (for normalisation)
        H_cd, _, _ = np.histogram2d(
            pixel_positions[mask, i1],
            pixel_positions[mask, i2],
            bins=pix,
            range=im_range,
            weights=10.0 ** dx[mask],
        )
        # Upsample to the finest resolution if needed
        if lvl < lmax:
            factor = 2 ** (lmax - lvl)
            H = H.repeat(factor, axis=0).repeat(factor, axis=1)
            H_cd = H_cd.repeat(factor, axis=0).repeat(factor, axis=1)
        image += H
        image_cd += H_cd
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        image = np.where(image_cd > 0, image / image_cd, 0)
    return image, image_cd

# ---------------------------------------------------------------
# Main analysis / plotting routine
# ---------------------------------------------------------------

def main():
    # -----------------------------------------------------------------
    # 1. Paths – adjust if your folder structure changes
    # -----------------------------------------------------------------
    base_path = '/Users/autumn/Documents/GitHub/hathor/test_data'
    gas_path = os.path.join(base_path, 'halo_1411_gas.bin')
    star_path = os.path.join(base_path, 'halo_1411_stars.bin')

    # -----------------------------------------------------------------
    # 2. Load data
    # -----------------------------------------------------------------
    print('Reading gas cutout ...')
    gas = read_megatron_cutout(gas_path)
    print('Reading star cutout ...')
    stars = read_megatron_star_cutout(star_path)

    # -----------------------------------------------------------------
    # 3. Compute a simple metallicity proxy for each gas cell
    # -----------------------------------------------------------------
    metal_cols = ["nC", "nN", "nO", "nNe", "nMg", "nSi", "nS", "nFe"]
    # Some columns may be missing depending on the file – keep the ones that exist
    available = [c for c in metal_cols if c in gas.columns]
    gas['Zproxy'] = gas[available].sum(axis=1)

    # -----------------------------------------------------------------
    # 4. Derive AMR level from dx (same recipe used in the example)
    # -----------------------------------------------------------------
    min_dx = gas['dx'].min()
    level = np.rint(20.0 - np.log2(10.0 ** gas['dx'] / (10.0 ** min_dx))).astype(int)
    gas['level'] = level

    # -----------------------------------------------------------------
    # 5. Project metallicity (z‑axis view)
    # -----------------------------------------------------------------
    pos = gas[['x', 'y', 'z']].values
    Z_image, _ = make_image(
        positions=pos,
        levels=gas['level'].values,
        feature=gas['Zproxy'].values,
        dx=gas['dx'].values,
        lmin=13,
        lmax=18,
        npix=1024,
        redshift=gas['redshift'].mean(),
        boxsize=50.0,  # you can change this if you know the real box size
        view_dir='z',
    )

    # -----------------------------------------------------------------
    # 6. Star selection – young (<50 Myr) and SFR proxy
    # -----------------------------------------------------------------
    # Assuming the age column is in Myr (common in many cutouts)
    young_mask = stars['age'] < 50.0
    young_stars = stars[young_mask].copy()
    # Specific SFR proxy: mass / age (avoid divide‑by‑zero)
    young_stars['sfr_proxy'] = young_stars['mass'] / np.maximum(young_stars['age'], 1e-3)

    # Project star positions onto the same plane (z view)
    star_pos = young_stars[['x', 'y', 'z']].values
    # Convert to pixel coordinates using the same centre/scale as the image
    # Re‑use the centre we computed inside make_image (median of gas positions)
    # For simplicity we recompute it here:
    cen_pix = np.median(pos * (2 ** 18), axis=0)
    star_pix = star_pos * (2 ** 18) - cen_pix + 512  # shift to image centre (1024/2)

    # -----------------------------------------------------------------
    # 7. Estimate halo centre and virial radius (Rvir)
    # -----------------------------------------------------------------
    gas_center = np.median(pos, axis=0)
    distances = np.linalg.norm(pos - gas_center, axis=1)
    Rvir = np.percentile(distances, 95)  # 95th percentile as a proxy
    print(f'Estimated Rvir ≈ {Rvir:.3f} (code units)')

    # Convert Rvir to pixel radius for the plot
    physical_boxsize = 50.0 * 1000.0 / (1.0 + gas['redshift'].mean())  # pkpc
    pkpc_per_pix = physical_boxsize / 1024.0
    Rvir_pix = Rvir / pkpc_per_pix

    # -----------------------------------------------------------------
    # 8. Plotting
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        Z_image,
        origin='lower',
        cmap='viridis',
        norm=LogNorm(vmin=Z_image[Z_image > 0].min(), vmax=Z_image.max()),
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Metallicity proxy (arbitrary units)')

    # Contours at 25, 50, 75 percentiles of the projected metallicity
    levels = np.percentile(Z_image[Z_image > 0], [25, 50, 75])
    cs = ax.contour(Z_image, levels=levels, colors='white', linewidths=0.8)
    ax.clabel(cs, fmt='%.2g', colors='white')

    # Overlay young stars, colour‑coded by sfr_proxy
    sc = ax.scatter(
        star_pix[:, 0], star_pix[:, 1],
        c=young_stars['sfr_proxy'],
        cmap='plasma',
        edgecolor='k',
        s=30,
        label='Stars <50 Myr',
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('SFR proxy (M$_\odot$ Myr$^{-1}$)')

    # Circle for Rvir
    circ = Circle((512, 512), Rvir_pix, edgecolor='red', facecolor='none', lw=2, ls='--', label='R$_{vir}$')
    ax.add_patch(circ)

    # Concentric annuli (0.2–1.0 Rvir in steps of 0.2)
    for f in np.arange(0.2, 1.01, 0.2):
        ann = Circle((512, 512), f * Rvir_pix, edgecolor='gray', facecolor='none', lw=0.5, ls=':', alpha=0.7)
        ax.add_patch(ann)

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Projected CGM Metallicity with Young Stars (age < 50 Myr)')
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()