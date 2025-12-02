import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import FortranFile
import pandas as pd

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
    df = pd.DataFrame(all_data, columns=header)

    # electron density
    ne = (10.**df["nH"]) * df["H_II"]
    ne += ((0.24*(10.**df["nH"])/0.76)/4.0) * (df["He_II"] + 2.0*df["He_III"])
    df["ne"] = ne

    # AMR level assignment
    tmp_levels = np.arange(31)
    mpc_2_cm = 3.086e+24
    loc_z = df["redshift"].mean()
    levels_dx = np.log10((((20.0/0.672699966430664) / (1. + loc_z)) / (2.**tmp_levels) * mpc_2_cm))
    cell_lengths = np.unique(df["dx"])
    level_arr = -999*np.ones(len(df), dtype=int)
    for cl in cell_lengths:
        idx = np.abs(cl - levels_dx).argmin()
        level = tmp_levels[idx]
        level_arr[df["dx"] == cl] = level
    df["level"] = level_arr
    return df

def make_image(positions, levels, features, dx, view_dir='z', npix=512,
               lmin=12, lmax=18, redshift=0.5, boxsize=20.0):
    physical_boxsize = boxsize * 1000.0 / (1.0 + redshift)  # pkpc
    width = (npix / (2**lmax)) * physical_boxsize
    pixel_positions = positions * (2**lmax)

    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_mean = 0.5 * (pos_min + pos_max)
    xcen, ycen, zcen = np.rint(pos_mean * (2**lmax))
    pp = np.arange(2**lmin) * (2**lmax) / (2**lmin)
    xcen = pp[np.argmin(np.abs(xcen - pp))]
    ycen = pp[np.argmin(np.abs(ycen - pp))]
    zcen = pp[np.argmin(np.abs(zcen - pp))]

    x_l, x_h = xcen - npix/2, xcen + npix/2
    y_l, y_h = ycen - npix/2, ycen + npix/2
    z_l, z_h = zcen - npix/2, zcen + npix/2

    if view_dir == 'x':
        im_range = ((y_l, y_h), (z_l, z_h))
        i1, i2 = 1, 2
    elif view_dir == 'y':
        im_range = ((x_l, x_h), (z_l, z_h))
        i1, i2 = 0, 2
    else:  # 'z'
        im_range = ((x_l, x_h), (y_l, y_h))
        i1, i2 = 0, 1

    l_pix_per_level = [int(npix / (2.**(lmax - l))) for l in range(lmin, lmax+1)]
    image = np.zeros((npix, npix))
    for idx, l in enumerate(range(lmin, lmax+1)):
        pix_per = l_pix_per_level[idx]
        if pix_per < 1:
            continue
        filt = levels == l
        H, _, _ = np.histogram2d(
            pixel_positions[:, i1][filt],
            pixel_positions[:, i2][filt],
            bins=pix_per,
            range=im_range,
            weights=features[filt] * (10.0**dx[filt])
        )
        if l < lmax:
            up = int(2**(lmax - l))
            H = H.repeat(up, axis=0).repeat(up, axis=1)
        image += H
    return image, None

if __name__ == "__main__":
    path = os.path.expanduser("~/Documents/GitHub/hathor/dataset_examples/halo_3517_gas.bin")
    with FortranFile(path, 'r') as ff:
        df = read_megatron_cutout(ff)

    positions = df[["x", "y", "z"]].values.astype(float)
    levels = df["level"].values.astype(int)
    features = df["ne"].values.astype(float)
    dx = df["dx"].values.astype(float)

    mask = np.isfinite(features) & np.isfinite(levels) & np.isfinite(dx)
    positions, levels, features, dx = positions[mask], levels[mask], features[mask], dx[mask]

    image, _ = make_image(
        positions,
        levels,
        features,
        dx,
        view_dir='z',
        npix=512,
        lmin=12,
        lmax=18,
        redshift=0.5,
        boxsize=20.0
    )

    os.makedirs("./figs", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(image, norm=LogNorm(), cmap="magma", origin="lower")
    plt.colorbar(label="Electron density (arbitrary units)")
    plt.xlabel("x [pkpc]")
    plt.ylabel("y [pkpc]")
    plt.title("Projected Electron Density")
    plt.tight_layout()
    plt.savefig("./figs/output_plot.png", dpi=300)
    plt.close()