import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.io import FortranFile
from scipy.interpolate import interp1d

# Plotting style
# plt.rc('text',usetex=True)
# font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
# plt.rc('font',**font)

"""
Functions for reading in cutouts
"""

elements = ["Fe","O","N","C","Mg","Ne","Si","S","Ca","CO"]
dep_solar = [0.01, 0.728, 0.603, 0.501, 0.163, 1.0, 0.1, 1.0, 0.003, 1.0]
#dep_solar = [0.01, 0.728, 0.603, 0.163, 0.100, 1.0, 0.003, 0.501, 1.0, 1.0]
def get_all_dust_depletions(nH,nO,T):
    #
    # Calculates the expected dust depletion
    #
    # ! Get the depletion factors
    # ! RR14 is based on the BARE-GR-S model of Zubko et al 2004
    # ! See Table 5 which is where we generatred the fractional contributions 
    # ! of each element
    # Note that this assumes all ingested quantities are in log

    depletion_table = np.ones((len(nH),len(elements)))
    filt = T < 6.0

    for i,(el,d) in enumerate(zip(elements,dep_solar)):
        x = 12.0 + (nO - nH)

        # Broken powerlaw model from RR14 (consistent with Taysun's Lya feedback)
        # See Table 1 of: https://www.aanda.org/articles/aa/pdf/2014/03/aa22803-13.pdf
        # We use the XCO,Z case
        a  = 2.21
        aH = 1.00
        b  = 0.96
        aL = 3.10
        xt = 8.10
        xs = 8.69

        y = a + (aH * (xs - x))
        y[x<=xt] = b + (aL * (xs - x[x<=xt]))

        y = 10.0**y # This is the Gas to Dust mass ratio

        # Fill table with depletions
        depletion_table[:,i][filt] = 1.0 - ( (1.0 - d) * np.minimum(1.0,162.0/y[filt]) )

    columns = [f"{e}_dep" for e in elements]

    return pd.DataFrame(depletion_table, columns=columns)

def read_megatron_cutout(ff,lmax=20.0,boxsize=50.0,h=0.672699966430664,cool_heat=False):
    """
    Function to read in a megatron gas cutout
    """

    header = [
        "redshift",
        "dx",
        "x", "y", "z",
        "vx", "vy", "vz",
        "nH", "T", "P",
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

    if (cool_heat):
        header += ["cooling","heating"]

    # load in all of the quantities from the cutout
    for i,q in enumerate(header):
        # Read in the quantity
        dat = ff.read_reals("float64")

        # If first iteration create the data array
        if i == 0:
            all_dat = np.zeros((len(dat),len(header)))

        all_dat[:,i] = dat

    # Now create the pandas array
    df = pd.DataFrame(all_dat, columns=header)

    # Now compute some extra properties
    cell_mass = 1.6735575e-24 * ((10.0**df["nH"])/0.76) * ((10.**df["dx"])**3) / 1.9891e33
    df["mass"] = cell_mass

    # Compute the electron density
    edens = (10.**df["nH"]) * df["H_II"]  # Hydrogen
    edens += ((0.24*(10.**df["nH"])/0.76)/4.0) * (df["He_II"] + 2.0 * df["He_III"]) # Helium
    metals = ["C","N","O","Ne","Mg","Si","S","Fe"]
    roman_nums = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"]
    for m in metals:
        for i,r in enumerate(roman_nums):
            key = f"{m}_{r}"
            if key in df.keys():
                edens += (10.**df[f"n{m}"]) * (df[key]*i)
    df["ne"] = edens

    # Compute the level for each cell
    loc_z = df["redshift"].mean()
    tmp_levels = np.arange(31)
    mpc_2_cm = 3.086e+24
    levels_dx = np.log10((((boxsize/h) / (1. + loc_z)) / (2.**tmp_levels)) * mpc_2_cm) # log10 cm
    # Initialize the array
    my_initial_levels = -999*np.ones(len(df),dtype=int)
    # Get the unique cell sizes
    cell_lengths = np.unique(df["dx"])
    for cl in cell_lengths:
        idx_loc = np.abs(cl-levels_dx).argmin()
        level_loc = tmp_levels[idx_loc]
        filt = df["dx"] == cl
        my_initial_levels[filt] = level_loc
    df["level"] = my_initial_levels

    # get the helium number density for each cell
    df["nHe"] = np.log10(((0.24*(10.**df["nH"])/0.76)/4.0))

    # Finally get the depletions
    dep_table = get_all_dust_depletions(df["nH"],df["nO"],df["T"])

    # Append the depletions
    df = pd.concat([df, dep_table], axis=1)

    # Make H and He dep 1
    df["H_dep"] = 1.0
    df["He_dep"] = 1.0

    """ 
    # Multiply the ion densities by the depletion factors
    # (addition because we are in log space)
    for e in elements:
        df[f"n{e}"] += np.log10(dep_table[f"{e}_dep"])
    """
    return df

def read_megatron_star_cutout(ff):
    """
    Function to read in a megatron star cutout
    """

    header = [
            "x","y","z",
            "vx","vy","vz",
            "age",
            "met_Fe", "met_O", "met_N", "met_Mg", "met_Ne", "met_Si", "met_Ca", "met_C", "met_S",
            "initial_mass","mass",
            ]

    # First read the total number of stars
    nstars = ff.read_ints("int32")[0]

    # Initialize the data array
    all_dat = np.zeros((nstars,len(header)))

    # load in all of the quantities from the cutout
    if nstars > 0:
        for i,q in enumerate(header):
            # Read in the quantity
            dat = ff.read_reals("float64")
            all_dat[:,i] = dat

        # Now create the pandas array
        df = pd.DataFrame(all_dat, columns=header)
    else:
        df = pd.DataFrame([], columns=header)
    return df

"""
Functions for making images from the cutouts
"""
def make_image(positions,levels,features,dx,lmin=13,lmax=18,npix=128,redshift=6.0,boxsize=20.0,
               view_dir='z'):
    """
    Function to make image from cutouts
    Example:

    # Get the level of each gas cell
    level = np.rint(20.0 - np.log2((10.0**df["dx"])/(10.0**min(df["dx"])))).astype(int)

    a,h = make_image(df[["x","y","z"]],level,df["ne"],df["dx"],view_dir='y',npix=2048,lmin=12,lmax=20,redshift=13.323885,boxsize=50.0)
    """
    physical_boxsize = boxsize * 1000./(1.0+redshift)
    width = (npix/(2**lmax))*physical_boxsize
    print(f"Image width: {round(width,4)} pkpc")

    pixel_positions = np.array(positions*(2**lmax))

    # If the center is none, center on the feature
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_mean = 0.5 * (pos_min + pos_max)
    xcen,ycen,zcen = np.rint(pos_mean*(2**lmax))
    pp = np.arange(2**lmin)*(2**lmax)/(2**lmin)
    xcen = pp[np.argmin(np.abs(xcen-pp))]
    ycen = pp[np.argmin(np.abs(ycen-pp))]
    zcen = pp[np.argmin(np.abs(zcen-pp))]

    # Get the ranges for the image
    x_l = xcen - npix/2
    x_h = xcen + npix/2
    y_l = ycen - npix/2
    y_h = ycen + npix/2
    z_l = zcen - npix/2
    z_h = zcen + npix/2

    if view_dir == 'x':
        im_range = ((y_l,y_h),(z_l,z_h))
        i1 = 1
        i2 = 2
    if view_dir == 'y':
        im_range = ((x_l,x_h),(z_l,z_h))
        i1 = 0
        i2 = 2
    if view_dir == 'z':
        im_range = ((x_l,x_h),(y_l,y_h))
        i1 = 0
        i2 = 1

    l_pix_per_level = [int(npix/(2.**(lmax-l))) for l in range(lmin,lmax+1)]

    image = np.zeros((npix,npix))
    all_images = []
    for i,l in enumerate(range(lmin,lmax+1)):
        if l_pix_per_level[i] < 1:
            continue

        filt = levels == l

        H, _, _ = np.histogram2d(pixel_positions[:,i1][filt],
                                 pixel_positions[:,i2][filt],
                                 bins=l_pix_per_level[i],
                                 range=im_range,weights=features[filt]*(10.0**dx[filt]))

        all_images.append(H)
        if l < lmax:
            up_samp = int(2**(lmax-l))
            H = H.repeat(up_samp, axis=1).repeat(up_samp, axis=0)

        image += H

    return image,all_images

bpass_age = np.array([
    6.299     ,  6.31472241,  6.33044482,  6.34616722,  6.36188963, 
    6.37761204,  6.39333445,  6.40905686,  6.42477926,  6.44050167, 
    6.45622408,  6.47194649,  6.4876689 ,  6.5033913 ,  6.51911371, 
    6.53483612,  6.55055853,  6.56628094,  6.58200334,  6.59772575, 
    6.61344816,  6.62917057,  6.64489298,  6.66061538,  6.67633779, 
    6.6920602 ,  6.70778261,  6.72350502,  6.73922742,  6.75494983, 
    6.77067224,  6.78639465,  6.80211706,  6.81783946,  6.83356187, 
    6.84928428,  6.86500669,  6.8807291 ,  6.89645151,  6.91217391, 
    6.92789632,  6.94361873,  6.95934114,  6.97506355,  6.99078595, 
    7.00650836,  7.02223077,  7.03795318,  7.05367559,  7.06939799, 
    7.0851204 ,  7.10084281,  7.11656522,  7.13228763,  7.14801003, 
    7.16373244,  7.17945485,  7.19517726,  7.21089967,  7.22662207, 
    7.24234448,  7.25806689,  7.2737893 ,  7.28951171,  7.30523411, 
    7.32095652,  7.33667893,  7.35240134,  7.36812375,  7.38384615, 
    7.39956856,  7.41529097,  7.43101338,  7.44673579,  7.46245819, 
    7.4781806 ,  7.49390301,  7.50962542,  7.52534783,  7.54107023, 
    7.55679264,  7.57251505,  7.58823746,  7.60395987,  7.61968227, 
    7.63540468,  7.65112709,  7.6668495 ,  7.68257191,  7.69829431, 
    7.71401672,  7.72973913,  7.74546154,  7.76118395,  7.77690635, 
    7.79262876,  7.80835117,  7.82407358,  7.83979599,  7.85551839, 
    7.8712408 ,  7.88696321,  7.90268562,  7.91840803,  7.93413043, 
    7.94985284,  7.96557525,  7.98129766,  7.99702007,  8.01274247, 
    8.02846488,  8.04418729,  8.0599097 ,  8.07563211,  8.09135452, 
    8.10707692,  8.12279933,  8.13852174,  8.15424415,  8.16996656, 
    8.18568896,  8.20141137,  8.21713378,  8.23285619,  8.2485786 , 
    8.264301  ,  8.28002341,  8.29574582,  8.31146823,  8.32719064, 
    8.34291304,  8.35863545,  8.37435786,  8.39008027,  8.40580268, 
    8.42152508,  8.43724749,  8.4529699 ,  8.46869231,  8.48441472, 
    8.50013712,  8.51585953,  8.53158194,  8.54730435,  8.56302676, 
    8.57874916,  8.59447157,  8.61019398,  8.62591639,  8.6416388 , 
    8.6573612 ,  8.67308361,  8.68880602,  8.70452843,  8.72025084, 
    8.73597324,  8.75169565,  8.76741806,  8.78314047,  8.79886288, 
    8.81458528,  8.83030769,  8.8460301 ,  8.86175251,  8.87747492, 
    8.89319732,  8.90891973,  8.92464214,  8.94036455,  8.95608696, 
    8.97180936,  8.98753177,  9.00325418,  9.01897659,  9.034699  , 
    9.0504214 ,  9.06614381,  9.08186622,  9.09758863,  9.11331104, 
    9.12903344,  9.14475585,  9.16047826,  9.17620067,  9.19192308, 
    9.20764548,  9.22336789,  9.2390903 ,  9.25481271,  9.27053512, 
    9.28625753,  9.30197993,  9.31770234,  9.33342475,  9.34914716, 
    9.36486957,  9.38059197,  9.39631438,  9.41203679,  9.4277592 , 
    9.44348161,  9.45920401,  9.47492642,  9.49064883,  9.50637124, 
    9.52209365,  9.53781605,  9.55353846,  9.56926087,  9.58498328, 
    9.60070569,  9.61642809,  9.6321505 ,  9.64787291,  9.66359532, 
    9.67931773,  9.69504013,  9.71076254,  9.72648495,  9.74220736, 
    9.75792977,  9.77365217,  9.78937458,  9.80509699,  9.8208194 , 
    9.83654181,  9.85226421,  9.86798662,  9.88370903,  9.89943144, 
    9.91515385,  9.93087625,  9.94659866,  9.96232107,  9.97804348, 
    9.99376589, 10.00948829, 10.0252107 , 10.04093311, 10.05665552, 
    10.07237793, 10.08810033, 10.10382274, 10.11954515, 10.13526756, 
    10.15098997, 10.16671237, 10.18243478, 10.19815719, 10.2138796 , 
    10.22960201, 10.24532441, 10.26104682, 10.27676923, 10.29249164, 
    10.30821405, 10.32393645, 10.33965886, 10.35538127, 10.37110368, 
    10.38682609, 10.40254849, 10.4182709 , 10.43399331, 10.44971572, 
    10.46543813, 10.48116054, 10.49688294, 10.51260535, 10.52832776, 
    10.54405017, 10.55977258, 10.57549498, 10.59121739, 10.6069398 , 
    10.62266221, 10.63838462, 10.65410702, 10.66982943, 10.68555184, 
    10.70127425, 10.71699666, 10.73271906, 10.74844147, 10.76416388, 
    10.77988629, 10.7956087 , 10.8113311 , 10.82705351, 10.84277592, 
    10.85849833, 10.87422074, 10.88994314, 10.90566555, 10.92138796, 
    10.93711037, 10.95283278, 10.96855518, 10.98427759, 11.          
])

bpass_mass = np.array([
    2.46183676e+00,  2.38183871e+00,  2.32142087e+00,  2.25892573e+00, 
    2.20580167e+00,  2.15736199e+00,  2.11172033e+00,  2.07002252e+00, 
    2.03021919e+00,  1.99194348e+00,  1.95357099e+00,  1.92123980e+00, 
    1.88722021e+00,  1.85491494e+00,  1.82518321e+00,  1.79734984e+00, 
    1.76923533e+00,  1.74335360e+00,  1.71869148e+00,  1.69357745e+00, 
    1.67103314e+00,  1.64839777e+00,  1.62700384e+00,  1.60624790e+00, 
    1.58609501e+00,  1.56638377e+00,  1.54758053e+00,  1.52858777e+00, 
    1.51033956e+00,  1.49289776e+00,  1.47456340e+00,  1.45714746e+00, 
    1.44025547e+00,  1.42384038e+00,  1.40762798e+00,  1.39129504e+00, 
    1.37539557e+00,  1.36038851e+00,  1.34482450e+00,  1.32994246e+00, 
    1.31526847e+00,  1.30069400e+00,  1.28643951e+00,  1.27245403e+00, 
    1.25878332e+00,  1.24517106e+00,  1.23157899e+00,  1.21822506e+00, 
    1.20489271e+00,  1.19249856e+00,  1.18016214e+00,  1.16802396e+00, 
    1.15598341e+00,  1.14410856e+00,  1.13298102e+00,  1.12185348e+00, 
    1.11093964e+00,  1.10055118e+00,  1.09016273e+00,  1.07977427e+00, 
    1.06992710e+00,  1.06011269e+00,  1.05029829e+00,  1.04053285e+00, 
    1.03124719e+00,  1.02196153e+00,  1.01267587e+00,  1.00339021e+00, 
    9.93983115e-01,  9.84839960e-01,  9.76035292e-01,  9.70857960e-01, 
    9.59314068e-01,  9.49693732e-01,  9.41912396e-01,  9.32983656e-01, 
    9.23783685e-01,  9.15227770e-01,  9.06734524e-01,  8.99443672e-01, 
    8.90057362e-01,  8.82003378e-01,  8.74489799e-01,  8.66362631e-01, 
    8.58178490e-01,  8.49840675e-01,  8.41804043e-01,  8.34702237e-01, 
    8.28527356e-01,  8.20911847e-01,  8.13452478e-01,  8.05929559e-01, 
    7.98853873e-01,  7.90847628e-01,  7.83337492e-01,  7.75877479e-01, 
    7.69350076e-01,  7.62157262e-01,  7.54415134e-01,  7.46871478e-01, 
    7.39306367e-01,  7.31918636e-01,  7.24294610e-01,  7.17341972e-01, 
    7.09484690e-01,  7.02339032e-01,  6.95508190e-01,  6.88593763e-01, 
    6.81291232e-01,  6.73696836e-01,  6.65375968e-01,  6.58869259e-01, 
    6.53253920e-01,  6.46395045e-01,  6.39749379e-01,  6.33248931e-01, 
    6.26066284e-01,  6.19143796e-01,  6.12390143e-01,  6.05652058e-01, 
    5.98743112e-01,  5.91639080e-01,  5.84710994e-01,  5.78053714e-01, 
    5.72030168e-01,  5.65770361e-01,  5.59098494e-01,  5.52167852e-01, 
    5.45050534e-01,  5.38617145e-01,  5.32293288e-01,  5.25923731e-01, 
    5.19547418e-01,  5.13150275e-01,  5.06749102e-01,  5.00304210e-01, 
    4.93844759e-01,  4.87365599e-01,  4.80874130e-01,  4.74420445e-01, 
    4.68018536e-01,  4.61654314e-01,  4.55561176e-01,  4.49468038e-01, 
    4.43209939e-01,  4.36851114e-01,  4.30507197e-01,  4.24257145e-01, 
    4.18007093e-01,  4.11850754e-01,  4.05782809e-01,  3.99714865e-01, 
    3.93600000e-01,  3.87465737e-01,  3.81331475e-01,  3.75151432e-01, 
    3.68961160e-01,  3.62770889e-01,  3.56776325e-01,  3.50821420e-01, 
    3.44866514e-01,  3.38949445e-01,  3.33058710e-01,  3.27167975e-01, 
    3.21222397e-01,  3.14988729e-01,  3.08755061e-01,  3.02521392e-01, 
    2.97361742e-01,  2.91519808e-01,  2.85425179e-01,  2.79146139e-01, 
    2.73766096e-01,  2.67777128e-01,  2.62001170e-01,  2.55786267e-01, 
    2.50530892e-01,  2.45149418e-01,  2.39617727e-01,  2.33887381e-01, 
    2.28290508e-01,  2.22992567e-01,  2.17415722e-01,  2.11347576e-01, 
    2.04813629e-01,  1.99295213e-01,  1.93387316e-01,  1.87730364e-01, 
    1.81260275e-01,  1.75045053e-01,  1.70535852e-01,  1.64993115e-01, 
    1.59798188e-01,  1.54264837e-01,  1.46116552e-01,  1.42463802e-01, 
    1.38128321e-01,  1.32209659e-01,  1.26564391e-01,  1.21324958e-01, 
    1.15774865e-01,  1.10818359e-01,  1.06078748e-01,  1.01001911e-01, 
    9.59085964e-02,  9.04609633e-02,  8.39172758e-02,  7.74793335e-02, 
    7.35763129e-02,  6.90844530e-02,  6.41751394e-02,  5.97654675e-02, 
    5.52819497e-02,  5.06620618e-02,  4.74226719e-02,  4.48109861e-02, 
    4.21993002e-02,  3.89578850e-02,  3.54350685e-02,  3.16689637e-02, 
    2.75791224e-02,  2.35256381e-02,  1.95176506e-02,  1.55249376e-02, 
    1.15571293e-02,  7.56974222e-03,  3.52508327e-03, -5.17837053e-04, 
    -4.54896166e-03, -8.58008626e-03, -1.26466892e-02, -1.67150844e-02, 
    -2.07921199e-02, -2.48720226e-02, -2.89519454e-02, -3.30318875e-02, 
    -3.71150037e-02, -4.12093407e-02, -4.53036777e-02, -4.94033825e-02, 
    -5.35037564e-02, -5.76144592e-02, -6.17351285e-02, -6.58558356e-02, 
    -6.99769455e-02, -7.40980554e-02, -7.82344602e-02, -8.23807984e-02, 
    -8.65272338e-02, -9.06747538e-02, -9.48222738e-02, -9.89765261e-02, 
    -1.03137602e-01, -1.07298678e-01, -1.11469188e-01, -1.15641309e-01, 
    -1.19812509e-01, -1.23978501e-01, -1.28144493e-01, -1.32317360e-01, 
    -1.36501922e-01, -1.40686484e-01, -1.44866460e-01, -1.49042313e-01, 
    -1.53218165e-01, -1.57400597e-01, -1.61587475e-01, -1.65774352e-01, 
    -1.69959516e-01, -1.74143489e-01, -1.78327463e-01, -1.82516532e-01, 
    -1.86710880e-01, -1.90905228e-01, -1.95104535e-01, -1.99315139e-01, 
    -2.03525743e-01, -2.07737419e-01, -2.11983281e-01, -2.16229144e-01, 
    -2.20475006e-01, -2.24747294e-01, -2.29032221e-01, -2.33317148e-01, 
    -2.37619724e-01, -2.41978066e-01, -2.46336408e-01, -2.50694751e-01, 
    -2.55208733e-01, -2.59776366e-01, -2.64343998e-01, -2.69127691e-01, 
    -2.74451333e-01, -2.79774974e-01, -2.85407441e-01, -2.92223042e-01, 
    -2.99038642e-01, -3.09470715e-01, -3.23581998e-01, -3.51319367e-01  
])

age_mass_interp = interp1d(bpass_age,bpass_mass,bounds_error=False,fill_value=(bpass_mass[0],bpass_mass[-1]))

def get_surviving_mass_frac_kroupa(m,slope_high=-2.3):
    """
    Returns the fraction of the mass of the star particle that is still on the main sequence
    This assumes a kroupa IMF
    """
    # The equation has an undefined at exactly -2
    if slope_high == -2.0:
        slope_high = -2.000001

    m_low = 0.1
    m_high = 300.

    slope_low = -1.3

    pivot = 0.5

    rescale = (pivot**slope_low) / (pivot**slope_high) 

    below_pivot_mass = (1./(slope_low + 2)) * (pivot**(slope_low+2)-m_low**(slope_low+2))
    above_pivot_mass = rescale * (1./(slope_high + 2)) * (m_high**(slope_high+2)-pivot**(slope_high+2))

    total_mass = below_pivot_mass + above_pivot_mass

    # If the mass if above the pivot
    if m >= pivot:
        dead_mass = rescale * (1./(slope_high + 2)) * (m_high**(slope_high+2)-m**(slope_high+2))
    else:
        dead_mass = (1./(slope_low + 2)) * (pivot**(slope_low+2)-m**(slope_low+2))
        dead_mass += above_pivot_mass 

    return 1.0 - min(max(dead_mass / total_mass,0.0),1.)

def get_pop3_ageMyr(mass):
    """
    Calculates the main-sequence lifetime of popIII stars of a given mass
    returns the age in Myr
    """
    a_fit = [0.7595398e+00, -3.7303953e+00, 1.4031973e+00, -1.7896967e-01]

    age_gyr = 0.0
    logm = np.log10(mass)

    for i in range(len(a_fit)):
       age_gyr +=a_fit[i] * (logm**i)

    age_gyr = 10.0**age_gyr
    return age_gyr * 1000.0 # returns Myr

def get_luminous_stellar_mass(df_stars):
    """
    Returns the mass of the star particle that is currently on the main sequence
    
    Note that age is in log10 yr
    Mass is in log10 Msol
    """

    # Differentiate between pop II and pop III stars
    stellar_metal = (2.09 * df_stars["met_O"]) + (1.06 * df_stars["met_Fe"])
    p3_star_filt = (stellar_metal < 2.e-8) # Select Pop III stars
    p2_star_filt = (stellar_metal >= 2.e-8) # Select Pop II stars

    # Initialize the survival fraction
    surv_frac = np.ones(len(df_stars))

    # Work with Pop II stars
    loc_age = df_stars["age"][p2_star_filt]*1e6
    loc_age[loc_age < 1.0] = 1.0 # HK prevent nans in log 10
    ms_age = 10.**age_mass_interp(np.log10(loc_age))

    # Get the survival fraction
    surv_fraci_p2 = np.array([get_surviving_mass_frac_kroupa(m,slope_high=-2.3) for m in ms_age])

    surv_frac[p2_star_filt] = surv_fraci_p2

    # Work with Pop III stars
    surv_fraci_p3 = np.ones(p3_star_filt.sum())

    # Get the main_sequence lifietimes of the pop iii stars
    p3_lifetime = get_pop3_ageMyr(df_stars["initial_mass"][p3_star_filt])

    # Filter evolved Pop III stars
    surv_fraci_p3[df_stars["age"][p3_star_filt] >= p3_lifetime] = 0.0

    surv_frac[p3_star_filt] = surv_fraci_p3

    df_stars["mass_aml"] = df_stars["initial_mass"] * surv_frac

    return



