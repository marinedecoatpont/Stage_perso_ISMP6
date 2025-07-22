import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import sys
import importlib
import pandas as pd
import os

#load personal function
sys.path.append('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/Fonction')
import ISMIP_function as ismip
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

# Choice of the simulations
target_simu = 'IGE_ElmerIce'
target_exp = 'expAE05'
target_year = 2273

colors = ['indigo', 'rebeccapurple', 'darkviolet', 'mediumslateblue', 'mediumblue', 'royalblue', 'cornflowerblue', 'skyblue', 'lightseagreen', 'mediumspringgreen', 'mediumseagreen', 'seagreen', 'green', 'limegreen', 'yellowgreen', 'greenyellow', 'yellow', 'khaki', 'gold', 'goldenrod', 'orange', 'darkorange', 'peru', 'orangered', 'tomato', 'red', 'maroon', 'indianred']
shapes = ['^']*14 + ['D']*14
alphas = [0.4, 0.4, 0.4, 0.4, 0.4, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4] 
sizes = [65, 65, 65, 65, 65, 150, 65, 65, 65, 65, 65]

where = 'j'
region = 'Amundsen'
density_ice = 917

flux_amun = [np.full(11,np.nan)]
flux_mean = []
path_data = '/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/Flux'
#df = pd.read_csv(f'/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/Summary_{target_simu}_{target_year}.csv')
df = pd.read_csv('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/IGE_ElmerIce_IGE_ElmerIce_2273/min_RMSE_IGE_ElmerIce_2273_IGE_ElmerIce.csv')
paths_target = [
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year-5}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year-4}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year-3}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year-2}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year-1}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year+1}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year+2}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year+3}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year+4}.nc',
        f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year+5}.nc'
]
target_flux_plot = []

for path in paths_target:
    if os.path.exists(path):
        gl_flux = xr.open_dataset(path).ligroundf
        flux_add = (abs(gl_flux).sum(dim = ['x', 'y'], skipna = True) * density_ice)/1e12
        target_flux_plot.append(flux_add.values.item())
    else:
        target_flux_plot.append(np.nan)

years = [int(target_year) - 5, int(target_year) - 4, int(target_year) - 3, int(target_year) - 2, int(target_year) - 1, int(target_year), int(target_year) + 1, int(target_year) + 2, int(target_year) + 3, int(target_year) + 4, int(target_year) + 5]

#----------------------- PLOT ICE FLUX AT THE GROUNDING LINE FOR RSME MINIMUM -------------------------------
fig = plt.figure(figsize=(15, 9))
fig.subplots_adjust(right=0.75)
ax = fig.add_axes([0.1, 0.1, 0.3, 0.6])  # [left, bottom, width, height]



for i, row in df.iterrows():
    #simu = row['simulation']
    simu = 'IGE_ElmerIce'
    exp = row['experiment']
    year_simu = int(row['min_year'])
    year_index = year_simu - 2016
    print(f'{simu} {exp}')

    paths = [
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu-5}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu-4}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu-3}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu-2}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu-1}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu+1}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu+2}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu+3}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu+4}.nc',
                f'{path_data}/ligroundf_{simu}_{exp}_{year_simu+5}.nc'
        ]

        # Ouverture principale (on suppose que celui-ci existe)
    flux = []

    for path in paths:
        if os.path.exists(path):
            gl_flux = xr.open_dataset(path).ligroundf
            flux_add = (abs(gl_flux).sum(dim = ['x', 'y'], skipna = True) * density_ice)/1e12
            flux.append(flux_add.values.item())
        else:
            print('path does not exist')
            flux.append(np.nan)
    ax.plot(years, flux, linestyle='--', color=colors[i], alpha=0.5)
    ax.scatter(years, flux, label=f'{simu} {exp}, {year_simu}', color=colors[i], marker=shapes[i], alpha=alphas, s=sizes)

    flux_mean.append(flux[5])
#moyenne
mean = np.mean(flux_mean)
std = np.std(flux_mean)


# Plot target
print(target_year)
ax.plot(years, target_flux_plot, color='#7B4DAE')
ax.scatter(years, target_flux_plot, label=f'{target_simu}, {target_year}',color='#7B4DAE', s=sizes)
plt.errorbar(int(target_year), mean, yerr = std, fmt = 'o', color = 'dimgray', capsize = 5)


# Add titles to the legend
handles, labels = ax.get_legend_handles_labels()
simulation_legend_title = "Simulations"

# Place simulation legend at the top left
simulation_legend = ax.legend(handles, labels,
                               bbox_to_anchor=(1.05, 1.1), loc='upper left', fontsize=10, title=simulation_legend_title, frameon = False)
simulation_legend.get_title().set_fontsize(12)
simulation_legend.get_title().set_weight('bold')
simulation_legend.get_title().set_horizontalalignment('left')  # Align title to the left
ax.add_artist(simulation_legend)


ax.grid(axis='y', alpha=0.7)
ax.set_ylim(0, 100)
xticks = [years[0], years[5], years[-1]]  # t-5, t, t+5
xticklabels = ['t-5', 't', 't+5']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=18)
ax.tick_params(axis='y', which='major', labelsize=18)
ax.set_title(f'Ice flux at the grounding line in {region} \n {target_simu} {target_year}', fontsize=25, weight='bold')
ax.set_ylabel('Ice flux [Gt/yr]', fontsize=20)
ax.set_xlabel('Years', fontsize=20)

plt.savefig(f'Ice_flux_{target_simu}_{target_year}.png', dpi=300)
print(f'Saved figure for {target_simu} ice flux at the grounding line.')

print('Everything seems fine ~(°w°~)')
print('----- END OF PROGRAM -----')