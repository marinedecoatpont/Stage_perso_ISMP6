import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import sys
import importlib
import pandas as pd
import os
from sklearn.linear_model import LinearRegression


#load personal function
sys.path.append('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/Fonction')
import ISMIP_function as ismip
importlib.reload(ismip)


#----------------------- PLOT RMSE OVER THE ABSOLUTE DIFFERENCE BETWEEN COMPARISON AND TARGET ICE FLUX AT THE GROUNDING LINE -------------------------------
# AUTHOR: marine de coatpont
# May 26, 2025
# IGE / ISMIP6 internship
#
#
# PLEASE READ README FOR MORE INFORMATION ON THIS SCRIPT
#
#------------------------------------------------------------------------------------------------------------------------------------------

# Choice of the simulations
target_simu = 'IGE_ElmerIce'
target_exp = 'expAE05'
target_year = 2273

#df = pd.read_csv(f'/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/Summary_{target_simu}_{target_year}.csv')
df = pd.read_csv('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/IGE_ElmerIce_IGE_ElmerIce_2273/min_RMSE_IGE_ElmerIce_2273_IGE_ElmerIce.csv')
path_data = '/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/Flux'

colors = ['indigo', 'rebeccapurple', 'darkviolet', 'mediumslateblue', 'mediumblue', 'royalblue', 'cornflowerblue', 'skyblue', 'lightseagreen', 'mediumspringgreen', 'mediumseagreen', 'seagreen', 'green', 'limegreen', 'yellowgreen', 'greenyellow', 'yellow', 'khaki', 'gold', 'goldenrod', 'orange', 'darkorange', 'peru', 'orangered', 'tomato', 'red', 'maroon', 'indianred']


density_ice = 917

path_target = f'{path_data}/ligroundf_{target_simu}_{target_exp}_{target_year}.nc'
gl_flux = xr.open_dataset(path_target).ligroundf
flux_target = (abs(gl_flux).sum(dim = ['x', 'y'], skipna = True) * density_ice)/1e12
flux_target = float(flux_target.values.item())

all_delta_flux = []
all_rmse = []
fig = plt.figure(figsize=(15, 9))
fig.subplots_adjust(right=0.75)
ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])

for i, row in df.iterrows():
    flux = []
    simu = 'IGE_ElmerIce'
    #simu = row['simulation']
    exp = row['experiment']
    year_simu = int(row['min_year'])
    year_index = year_simu - 2016
    print(f'{simu} {exp}')

    path_rmse = f'/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/{target_simu}_{simu}_{target_year}/RMSE_{exp}.csv'
    df_rmse = pd.read_csv(path_rmse)
    rmse_values = df_rmse['rmse'].tolist()

    path = f'{path_data}/ligroundf_{simu}_{exp}.nc'
    flux_data = xr.open_dataset(path).ligroundf
    for year in range (len(flux_data.time)):
        gl_flux = flux_data.isel(time=year)
        flux_add = (abs(gl_flux).sum(dim = ['x', 'y'], skipna = True) * density_ice)/1e12
        flux.append(flux_add.values.item())

    rmse = np.array(rmse_values)
    flux = np.array(flux)
    delta_flux = np.abs(flux - flux_target)
    all_delta_flux.extend(delta_flux)
    all_rmse.extend(rmse)

    ax.scatter(delta_flux, rmse, color = colors[i], label = f'{simu} {exp}')


# Perform overall linear regression
all_delta_flux = np.array(all_delta_flux).reshape(-1, 1)
all_rmse = np.array(all_rmse)

reg = LinearRegression().fit(all_delta_flux, all_rmse)
slope = reg.coef_[0]
intercept = reg.intercept_

# Plot the overall regression line
x_vals = np.linspace(min(all_delta_flux), max(all_delta_flux), 100).reshape(-1, 1)
y_vals = reg.predict(x_vals)

ax.plot(x_vals, y_vals, color='grey', linestyle='-', label=f'Overall Regression\n(slope={slope:.6f}, intercept={intercept:.6f})')
ax.set_xlabel(r'$\Delta F = |F_{comparison} - F_{target}|$', fontsize = 20)
ax.set_ylabel('RMSE', fontsize = 20)
ax.set_title(f'RMSE over relative ice flux at the grounding line in Amundsen \n {target_simu} {target_year}', fontsize = 25, weight='bold')
handles, labels = ax.get_legend_handles_labels()
simulation_legend_title = "Simulations"

# Place simulation legend at the top left
simulation_legend = ax.legend(handles, labels,
                               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title=simulation_legend_title, frameon = False)
simulation_legend.get_title().set_fontsize(12)
simulation_legend.get_title().set_weight('bold')
simulation_legend.get_title().set_horizontalalignment('left')  # Align title to the left
ax.add_artist(simulation_legend)
fig.savefig(f'RMSE_flux_{target_simu}_{target_year}.png', dpi=300)

