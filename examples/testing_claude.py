import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Paths to data
gas_file = "/Users/autumn/Documents/GitHub/hathor/test_data/halo_1411_gas.bin"
stars_file = "/Users/autumn/Documents/GitHub/hathor/test_data/halo_1411_stars.bin"

# Load gas data
gas_data = np.fromfile(gas_file, dtype=np.float64)
n_gas_fields = 12
n_gas = len(gas_data) // n_gas_fields
gas_data = gas_data.reshape(n_gas, n_gas_fields)

# Extract gas properties
gas_x, gas_y, gas_z = gas_data[:, 0], gas_data[:, 1], gas_data[:, 2]
gas_vx, gas_vy, gas_vz = gas_data[:, 3], gas_data[:, 4], gas_data[:, 5]
gas_rho = gas_data[:, 6]

# Ion abundances for ionization state
gas_HI = gas_data[:, 7]
gas_HII = gas_data[:, 8]
ionization_fraction = gas_HII / (gas_HI + gas_HII + 1e-20)

# Load stellar data
stars_data = np.fromfile(stars_file, dtype=np.float64)
n_star_fields = 8
n_stars = len(stars_data) // n_star_fields
stars_data = stars_data.reshape(n_stars, n_star_fields)

stars_x, stars_y, stars_z = stars_data[:, 0], stars_data[:, 1], stars_data[:, 2]
stars_vx, stars_vy, stars_vz = stars_data[:, 3], stars_data[:, 4], stars_data[:, 5]
stars_mass = stars_data[:, 6]

# Calculate halo center (center of mass)
halo_center = np.array([
    np.average(stars_x, weights=stars_mass),
    np.average(stars_y, weights=stars_mass),
    np.average(stars_z, weights=stars_mass)
])

# Calculate halo bulk velocity
halo_velocity = np.array([
    np.average(stars_vx, weights=stars_mass),
    np.average(stars_vy, weights=stars_mass),
    np.average(stars_vz, weights=stars_mass)
])

# Gas positions and velocities relative to halo
gas_pos = np.column_stack([gas_x, gas_y, gas_z])
gas_vel = np.column_stack([gas_vx, gas_vy, gas_vz])
gas_pos_rel = gas_pos - halo_center
gas_vel_rel = gas_vel - halo_velocity

# Calculate distances from halo center
gas_r = np.linalg.norm(gas_pos_rel, axis=1)

# Calculate radial velocity (positive = outflow, negative = inflow)
gas_r_hat = gas_pos_rel / (gas_r[:, np.newaxis] + 1e-10)  # Unit vector
gas_v_radial = np.sum(gas_vel_rel * gas_r_hat, axis=1)

# Calculate tangential velocity
gas_v_tangential = np.sqrt(np.sum(gas_vel_rel**2, axis=1) - gas_v_radial**2)

# Classify flows
outflow_mask = gas_v_radial > 50  # km/s
inflow_mask = gas_v_radial < -50
rotating_mask = ~outflow_mask & ~inflow_mask

# Create figure
fig = plt.figure(figsize=(16, 10))

# Panel 1: 2D velocity field map (XY projection)
ax1 = fig.add_subplot(221)
# Color by radial velocity
scatter = ax1.scatter(gas_pos_rel[:, 0], gas_pos_rel[:, 1], 
                     c=gas_v_radial, s=3, cmap='RdBu_r',
                     vmin=-300, vmax=300, alpha=0.6, rasterized=True)
# Add velocity vectors (subsample for clarity)
subsample = np.random.choice(n_gas, size=min(500, n_gas), replace=False)
scale = 0.05  # Adjust arrow scale
ax1.quiver(gas_pos_rel[subsample, 0], gas_pos_rel[subsample, 1],
          gas_vel_rel[subsample, 0], gas_vel_rel[subsample, 1],
          alpha=0.3, scale=3000, width=0.003)
# Mark halo center
ax1.scatter([0], [0], c='yellow', s=200, marker='*', 
           edgecolors='black', linewidths=2, label='Halo center')
ax1.set_xlabel('X - X_center (kpc)')
ax1.set_ylabel('Y - Y_center (kpc)')
ax1.set_title('Gas Velocity Field (z=3.9)')
ax1.set_aspect('equal')
cbar1 = plt.colorbar(scatter, ax=ax1, label='Radial Velocity (km/s)\n(+: outflow, -: inflow)')
ax1.legend(loc='upper right')

# Panel 2: Radial velocity vs radius, colored by ionization
ax2 = fig.add_subplot(222)
scatter2 = ax2.scatter(gas_r, gas_v_radial, c=ionization_fraction, 
                      s=2, cmap='plasma', vmin=0, vmax=1, alpha=0.5, rasterized=True)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(y=100, color='red', linestyle=':', linewidth=1, alpha=0.5, label='v_r = +100 km/s')
ax2.axhline(y=-100, color='blue', linestyle=':', linewidth=1, alpha=0.5, label='v_r = -100 km/s')
ax2.set_xlabel('Radius from halo center (kpc)')
ax2.set_ylabel('Radial Velocity (km/s)')
ax2.set_title('Kinematics vs Radius (color = ionization)')
ax2.set_ylim(-400, 400)
ax2.legend()
ax2.grid(alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax2, label='HII/(HI+HII)')

# Panel 3: Radial velocity distribution
ax3 = fig.add_subplot(223)
v_bins = np.linspace(-500, 500, 60)
ax3.hist(gas_v_radial[outflow_mask], bins=v_bins, alpha=0.6, 
        color='red', label=f'Outflows (v_r > 50 km/s): {np.sum(outflow_mask)} cells', 
        edgecolor='black', linewidth=0.5)
ax3.hist(gas_v_radial[inflow_mask], bins=v_bins, alpha=0.6, 
        color='blue', label=f'Inflows (v_r < -50 km/s): {np.sum(inflow_mask)} cells',
        edgecolor='black', linewidth=0.5)
ax3.hist(gas_v_radial[rotating_mask], bins=v_bins, alpha=0.4, 
        color='gray', label=f'Rotating/tangential: {np.sum(rotating_mask)} cells',
        edgecolor='black', linewidth=0.5)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax3.axvline(x=np.median(gas_v_radial), color='green', linestyle='-', 
           linewidth=2, label=f'Median = {np.median(gas_v_radial):.1f} km/s')
ax3.set_xlabel('Radial Velocity (km/s)')
ax3.set_ylabel('Number of gas cells')
ax3.set_title('Radial Velocity Distribution')
ax3.legend(fontsize=8)
ax3.set_yscale('log')
ax3.grid(alpha=0.3)

# Panel 4: Phase diagram - radial velocity vs density
ax4 = fig.add_subplot(224)
log_rho = np.log10(gas_rho + 1e-30)
hist2d, xedges, yedges = np.histogram2d(
    log_rho, gas_v_radial, bins=[40, 40],
    range=[[np.percentile(log_rho, 1), np.percentile(log_rho, 99)], [-400, 400]]
)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax4.imshow(np.log10(hist2d.T + 1), origin='lower', aspect='auto',
               extent=extent, cmap='viridis', interpolation='nearest')
ax4.axhline(y=0, color='white', linestyle='--', linewidth=2, alpha=0.8)
ax4.set_xlabel('log₁₀(Gas Density) [code units]')
ax4.set_ylabel('Radial Velocity (km/s)')
ax4.set_title('Velocity-Density Phase Space')
ax4.text(0.05, 0.95, 'Outflows', transform=ax4.transAxes, 
        color='white', fontsize=10, verticalalignment='top', weight='bold')
ax4.text(0.05, 0.05, 'Inflows', transform=ax4.transAxes,
        color='white', fontsize=10, verticalalignment='bottom', weight='bold')
cbar4 = plt.colorbar(im, ax=ax4, label='log₁₀(Count)')

plt.tight_layout()
plt.savefig('kinematics_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved as kinematics_analysis.png")
print(f"\nFlow statistics:")
print(f"  Outflows: {np.sum(outflow_mask)} cells ({100*np.sum(outflow_mask)/n_gas:.1f}%)")
print(f"  Inflows: {np.sum(inflow_mask)} cells ({100*np.sum(inflow_mask)/n_gas:.1f}%)")
print(f"  Rotating/tangential: {np.sum(rotating_mask)} cells ({100*np.sum(rotating_mask)/n_gas:.1f}%)")
print(f"  Mean outflow velocity: {np.mean(gas_v_radial[outflow_mask]):.1f} km/s")
print(f"  Mean inflow velocity: {np.mean(gas_v_radial[inflow_mask]):.1f} km/s")
plt.show()