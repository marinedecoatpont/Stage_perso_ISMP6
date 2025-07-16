import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm
import sys
import importlib
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

#load personal function
sys.path.append('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/Fonction')
import Function.ISMIP_function as ismip
importlib.reload(ismip)

#---------------------------------- PLOT DIAGNOSTICS  -----------------------------------------------
#AUTHOR: marine de coatpont
#june 12, 2025
#IGE / ISMIP6 internship
#
#
#PLEASE READ README () FOR MORE INFORMATION ON THIS SCRIPT
#
#------------------------------------------------------------------------------------------------------------------------------------------

# Choice of the simulations
print('Enter the target simulation, experiment, and year for diagnostics plotting.')
target_simu = input("Enter the simulation you want to plot diagnostics for (e.g., DC_ISSM): ")
target_exp = input("Enter the experiment you want to plot diagnostics for (e.g., expAE04): ")
target_year = int(input("Enter the year you want to plot diagnostics for (e.g., '2020'): "))
year_index = target_year - 2016

print('Enter the comparison simulation, experiment, and year for diagnostics plotting.')
comp_simu = input("Enter the comparison simulation you want to plot diagnostics for (e.g., DC_ISSM): ")
comp_exp = input("Enter the comparison experiment you want to plot diagnostics for (e.g., expAE04): ")
comp_year = int(input("Enter the year you want to plot diagnostics for (e.g., '2020'): "))
comp_year_index = comp_year - 2016

condition = ['ULB_fETISh-KoriBU2', 'UNN_Ua']

color = ListedColormap(['wheat', 'lightblue'])
color.set_bad(color='gainsboro')

# Open the observation data for the target
grounded_data_target = xr.open_dataset(f'/home/jovyan/private-storage/result/{target_simu}/grounding_mask_{target_simu}_{target_exp}.nc')
grounded_interp_target = ismip.grid_4x4(grounded_data_target.grounding_mask.isel(time=year_index))

surface_target = ismip.open_file(target_simu, target_exp, 'orog')
surface_interp_target = ismip.grid_4x4(surface_target.orog.isel(time=year_index))

vx_data_target = ismip.open_file(target_simu, target_exp, 'xvelmean')
vx_interp_target = ismip.grid_4x4(vx_data_target.xvelmean.isel(time=year_index))

if target_simu in condition:
    vy_data_target = xr.open_dataset(f'/home/jovyan/private-storage/simu/{target_simu}/{target_exp}/yvelmean_AIS_{target_simu}_{target_exp}.nc', decode_times=False)
else:
    vy_data_target = ismip.open_file(target_simu, target_exp, 'yvelmean')
vy_interp_target = ismip.grid_4x4(vy_data_target.yvelmean.isel(time=year_index))

grounded_mask_target = ismip.amundsen_mask(grounded_interp_target)
vy_mask_target = ismip.amundsen_mask(vy_interp_target)
vy_mask_target = xr.where(grounded_mask_target == 0, vy_mask_target, np.nan)

vx_mask_target = ismip.amundsen_mask(vx_interp_target)
vx_mask_target = xr.where(grounded_mask_target == 0, vx_interp_target, np.nan)

surface_mask_target = ismip.amundsen_mask(surface_interp_target)
surface_mask_target = xr.where(grounded_mask_target == 0, surface_interp_target, np.nan)

# Open the observation data for the comparison
grounded_data_comp = xr.open_dataset(f'/home/jovyan/private-storage/result/{comp_simu}/grounding_mask_{comp_simu}_{comp_exp}.nc')
grounded_interp_comp = ismip.grid_4x4(grounded_data_comp.grounding_mask.isel(time=comp_year_index))

surface_comp = ismip.open_file(comp_simu, comp_exp, 'orog')
surface_interp_comp = ismip.grid_4x4(surface_comp.orog.isel(time=comp_year_index))

vx_data_comp = ismip.open_file(comp_simu, comp_exp, 'xvelmean')
vx_interp_comp = ismip.grid_4x4(vx_data_comp.xvelmean.isel(time=comp_year_index))

if comp_simu in condition:
    vy_data_comp = xr.open_dataset(f'/home/jovyan/private-storage/simu/{comp_simu}/{comp_exp}/yvelmean_AIS_{comp_simu}_{comp_exp}.nc', decode_times=False)
else:
    vy_data_comp = ismip.open_file(comp_simu, comp_exp, 'yvelmean')
vy_interp_comp = ismip.grid_4x4(vy_data_comp.yvelmean.isel(time=comp_year_index))

grounded_mask_comp = ismip.amundsen_mask(grounded_interp_comp)
vy_mask_comp = ismip.amundsen_mask(vy_interp_comp)
vy_mask_comp = xr.where(grounded_mask_comp == 0, vy_mask_comp, np.nan)

vx_mask_comp = ismip.amundsen_mask(vx_interp_comp)
vx_mask_comp = xr.where(grounded_mask_comp == 0, vx_interp_comp, np.nan)

surface_mask_comp = ismip.amundsen_mask(surface_interp_comp)
surface_mask_comp = xr.where(grounded_mask_comp == 0, surface_interp_comp, np.nan)
    

# Calculate the velocity norm for the target simulation
vitesse_target = np.sqrt(vx_mask_target**2 + vy_mask_target**2)
vitesse_target = vitesse_target * 365.25 * 24 * 3600

# Calculate the velocity norm for the comparison simulation
vitesse_comp = np.sqrt(vx_mask_comp**2 + vy_mask_comp**2)
vitesse_comp = vitesse_comp * 365.25 * 24 * 3600

# Set the color scale limits
vmin = 1e-1
vmax = 1e3
#vmax = 1e4 #UNN_Ua
norm = LogNorm(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(3, 3, figsize=(36, 30))
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((5, 6))

# First row: Target
fig.text(0.01, 0.85, f'{target_simu} {target_exp} {target_year}', va='center', rotation='vertical', fontsize=30, fontweight='bold')
pcm1 = axes[0, 0].pcolormesh(grounded_mask_target.x, grounded_mask_target.y, grounded_mask_target, cmap=color, vmin=0, vmax=2, shading='auto')
handles = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='wheat', markersize=10, label='Grounded area'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=10, label='Ice shelf / Open ocean'),
]
axes[0, 0].legend(handles=handles, loc='upper right', fontsize=14)
axes[0, 0].set_ylim(-1.0e6, 0.25e6)
axes[0, 0].set_xlim(-2.0e6, -0.75e6)
axes[0, 0].tick_params(axis='both', which='major', labelsize=20)
axes[0, 0].set_aspect('equal')
axes[0, 0].set_title("Grounded mask", fontsize=25, fontweight='bold')

pcm2 = axes[0, 1].pcolormesh(surface_mask_target.x, surface_mask_target.y, surface_mask_target, cmap='YlGnBu_r', vmin=0, vmax=3500, shading='auto')
axes[0, 1].set_ylim(-1.0e6, 0.25e6)
axes[0, 1].set_xlim(-2.0e6, -0.75e6)
axes[0, 1].tick_params(axis='both', which='major', labelsize=20)
axes[0, 1].set_aspect('equal')
axes[0, 1].set_title("Surface elevation", fontsize=25, fontweight='bold')
cbar2 = fig.colorbar(pcm2, ax=axes[0, 1])
cbar2.ax.set_title("Surface [m]", fontsize=18)
cbar2.ax.tick_params(labelsize=16)

pcm3 = axes[0, 2].pcolormesh(vitesse_target.x, vitesse_target.y, vitesse_target, cmap='Reds', norm=norm, shading='auto')
axes[0, 2].set_ylim(-1.0e6, 0.25e6)
axes[0, 2].set_xlim(-2.0e6, -0.75e6)
axes[0, 2].tick_params(axis='both', which='major', labelsize=20)
axes[0, 2].set_aspect('equal')
axes[0, 2].set_title("Velocity norm (log scale)", fontsize=25, fontweight='bold')
cbar3 = fig.colorbar(pcm3, ax=axes[0, 2], format=formatter)
cbar3.ax.set_title("Velocity [m/yr]", fontsize=18)
cbar3.ax.tick_params(labelsize=16)

# Second row: Comparison
fig.text(0.01, 0.5, f'{comp_simu} {comp_exp} {comp_year}', va='center', rotation='vertical', fontsize=30, fontweight='bold')
pcm4 = axes[1, 0].pcolormesh(grounded_mask_comp.x, grounded_mask_comp.y, grounded_mask_comp, cmap=color, vmin=0, vmax=2, shading='auto')
axes[1, 0].set_ylim(-1.0e6, 0.25e6)
axes[1, 0].set_xlim(-2.0e6, -0.75e6)
axes[1, 0].tick_params(axis='both', which='major', labelsize=20)
axes[1, 0].set_aspect('equal')

pcm5 = axes[1, 1].pcolormesh(surface_mask_comp.x, surface_mask_comp.y, surface_mask_comp, cmap='YlGnBu_r', vmin=0, vmax=3500, shading='auto')
axes[1, 1].set_ylim(-1.0e6, 0.25e6)
axes[1, 1].set_xlim(-2.0e6, -0.75e6)
axes[1, 1].tick_params(axis='both', which='major', labelsize=20)
axes[1, 1].set_aspect('equal')
cbar5 = fig.colorbar(pcm5, ax=axes[1, 1])
cbar5.ax.set_title("Surface [m]", fontsize=18)
cbar5.ax.tick_params(labelsize=16)

pcm6 = axes[1, 2].pcolormesh(vitesse_comp.x, vitesse_comp.y, vitesse_comp, cmap='Reds', norm=norm, shading='auto')
axes[1, 2].set_ylim(-1.0e6, 0.25e6)
axes[1, 2].set_xlim(-2.0e6, -0.75e6)
axes[1, 2].tick_params(axis='both', which='major', labelsize=20)
axes[1, 2].set_aspect('equal')
cbar6 = fig.colorbar(pcm6, ax=axes[1, 2], format=formatter)
cbar6.ax.set_title("Velocity [m/yr]", fontsize=18)
cbar6.ax.tick_params(labelsize=16)

# Third row: Differences
fig.text(0.01, 0.15, "Target - Comparison", va='center', rotation='vertical', fontsize=30, fontweight='bold')
diff_grounded = grounded_mask_target - grounded_mask_comp
color_diff = ListedColormap(['darkblue', 'antiquewhite', 'brown'])
pcm7 = axes[2, 0].pcolormesh(diff_grounded.x, diff_grounded.y, diff_grounded, cmap=color_diff, vmin=-1, vmax=1, shading='auto')
axes[2, 0].set_ylim(-1.0e6, 0.25e6)
axes[2, 0].set_xlim(-2.0e6, -0.75e6)
axes[2, 0].tick_params(axis='both', which='major', labelsize=20)
axes[2, 0].set_aspect('equal')
axes[2, 0].set_title("Grounded mask difference", fontsize=25, fontweight='bold')

# Add legend for the color mapping
handles_diff = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue', markersize=10, label='Floating on comparison'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='antiquewhite', markersize=10, label='Same state'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='brown', markersize=10, label='Floating on target'),
]
axes[2, 0].legend(handles=handles_diff, loc='upper right', fontsize=14)

diff_surface = surface_mask_target - surface_mask_comp
pcm8 = axes[2, 1].pcolormesh(diff_surface.x, diff_surface.y, diff_surface, cmap='coolwarm', shading='auto', vmin=-np.nanmax(np.abs(diff_surface)), vmax=np.nanmax(np.abs(diff_surface)))
axes[2, 1].set_ylim(-1.0e6, 0.25e6)
axes[2, 1].set_xlim(-2.0e6, -0.75e6)
axes[2, 1].tick_params(axis='both', which='major', labelsize=20)
axes[2, 1].set_aspect('equal')
axes[2, 1].set_title("Surface elevation difference", fontsize=25, fontweight='bold')
cbar8 = fig.colorbar(pcm8, ax=axes[2, 1])
cbar8.ax.set_title("Difference [m]", fontsize=18)
cbar8.ax.tick_params(labelsize=16)

diff_vitesse = vitesse_target - vitesse_comp
pcm9 = axes[2, 2].pcolormesh(diff_vitesse.x, diff_vitesse.y, diff_vitesse, cmap='coolwarm', shading='auto', vmin=-np.nanmax(np.abs(diff_vitesse)), vmax=np.nanmax(np.abs(diff_vitesse)))
axes[2, 2].set_ylim(-1.0e6, 0.25e6)
axes[2, 2].set_xlim(-2.0e6, -0.75e6)
axes[2, 2].tick_params(axis='both', which='major', labelsize=20)
axes[2, 2].set_aspect('equal')
axes[2, 2].set_title("Velocity norm difference", fontsize=25, fontweight='bold')
cbar9 = fig.colorbar(pcm9, ax=axes[2, 2])
cbar9.ax.set_title("Difference [m/yr]", fontsize=18)
cbar9.ax.tick_params(labelsize=16)

plt.tight_layout()
fig.savefig(f'{target_simu}_{comp_simu}_{comp_exp}_comparison_diag.png', dpi=300)
print(f'Saved comparison figure for {target_simu} {target_exp} {target_year}.')

print('Everything seems fine ~(°w°~)')
print('----- END OF PROGRAM -----')