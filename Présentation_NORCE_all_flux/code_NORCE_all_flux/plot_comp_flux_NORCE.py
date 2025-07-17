import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import sys
import importlib
import pandas as pd
import os
import config


#load personal function
sys.path.append(f'{config.SAVE_PATH}/Function')
import Function.ISMIP_function as ismip
importlib.reload(ismip)


#----------------------- PLOT ICE FLUX AT THE GROUNDING LINE FOR RSME MINIMUM -------------------------------
# AUTHOR: marine de coatpont
# April 18, 2025
# IGE / ISMIP6 internship
#
#
#PLEASE READ README FOR MORE INFORMATION ON THIS SCRIPT
#
#------------------------------------------------------------------------------------------------------------------------------------------

experiment = ['expAE01', 'expAE02', 'expAE03','expAE04', 'expAE05', 'expAE06']
colors = ['darkblue', 'darkturquoise', 'darkorange', 'mediumaquamarine', 'limegreen', 'tomato']
shapes = ['^', '^', '^', '^', '^', '^']
alphas = [0.4] * 6
sizes = [65] * 6

flux_mean = []
target_year = 2250
df_path = '/Users/marine/Documents/Master/M2/Stage/code_openreprolab/Result/min_RMSE_UNN_Ua_2250_NORCE_CISM2-MAR364-ERA-t1.csv'
df = pd.read_csv(df_path)

# Plot
fig = plt.figure(figsize=(15, 9))
ax = fig.add_axes([0.1, 0.1, 0.3, 0.6])  # [left, bottom, width, height]

for index, row in df.iterrows():
    simu = row['simulation']
    exp = row['experiment']
    year_simu = int(row['min_year'])

    # Années autour de l'année minimale
    years = list(range(target_year - 5, target_year + 6))
    
    # Construction des paths
    paths = [
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu-5}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu-4}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu-3}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu-2}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu-1}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu+1}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu+2}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu+3}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu+4}.nc',
                f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{year_simu+5}.nc'
               ]
    
    flux = []
    print(f"\nSimulation: {simu}, Exp: {exp}, Year: {year_simu}")

    for path in paths:
        if os.path.exists(path):
            flux_add= ismip.basin_flux_hand(xr.open_dataset(path).ligroundf, config.REGION)
            flux.append(flux_add)
        else:
            flux.append(np.nan)

    if not all(np.isnan(flux)):
        c = colors[index % len(colors)]
        m = shapes[index % len(shapes)]
        a = alphas[index % len(alphas)]
        s = sizes[index % len(sizes)]
        
        ax.plot(years, flux, linestyle='--', color=c, alpha=0.5)
        ax.scatter(years, flux, label=f'{simu} {exp}, {year_simu}', color=c, marker=m, alpha=a, s=s)
        flux_mean.append(flux[5])
    else:
        print(f"Aucune donnée valide pour {simu} {exp}")

# Ajouter les flux en violet pour UNN Ua expAE04 2250
target_simu = 'UNN_Ua'
target_exp = 'expAE04'
target_year_simu = 2250
target_paths = [
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu-5}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu-4}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu-3}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu-2}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu-1}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu+1}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu+2}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu+3}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu+4}.nc',
    f'{config.PATH_IF}/{target_simu}/ligroundf_{target_simu}_{target_exp}_{target_year_simu+5}.nc'
]

target_flux = []
for path in target_paths:
    if os.path.exists(path):
        flux_add = ismip.basin_flux_hand(xr.open_dataset(path).ligroundf, config.REGION)
        target_flux.append(flux_add)
    else:
        target_flux.append(np.nan)
ax.plot(years, target_flux, linestyle='-', color='purple', alpha=0.8, linewidth=2)
ax.scatter(years, target_flux, label=f'{target_simu} {target_exp}, {target_year_simu}', color='purple', marker='o', alpha=0.8, s=80)

# Mise en forme
ax.grid(axis='y', alpha=0.7)
xticks = [years[0], years[5], years[-1]]
xticklabels = ['t-5', 't', 't+5']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=18)
ax.tick_params(axis='y', which='major', labelsize=18)
ax.set_title(f'Ice flux at the grounding line \n {config.REGION}', fontsize=25, weight='bold')
ax.set_ylabel('Ice flux [Gt/yr]', fontsize=20)
ax.set_xlabel('Years', fontsize=20)

# Légende
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Simulations", fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))

# Sauvegarde
plt.tight_layout()
save_path = f'{config.PATH_IF}/Ice_flux_NORCE3.png'
plt.savefig(save_path, dpi=300)
print(f"Figure saved: {save_path}")