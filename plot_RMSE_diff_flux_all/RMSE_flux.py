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
from sklearn.linear_model import LinearRegression


#load personal function
sys.path.append(f'{config.SAVE_PATH}/Function')
import Function.ISMIP_function as ismip
importlib.reload(ismip)


#----------------------- PLOT RMSE OVER THE ABSOLUTE DIFFERENCE BETWEEN COMPARISON AND TARGET ICE FLUX AT THE GROUNDING LINE -------------------------------
# AUTHOR: marine de coatpont
# May 26, 2025
# IGE / ISMIP6 internship
#
# This script plot the RMSE over the absolute difference between comparison and target ice flux at the grounding line for the comparison done in the report.
# It plots the scatter for each models used as comparison and the linear regression for the different scenaros.
# Tho colors and the shape of the scatter are the differente scenarios.
#
# PLEASE READ README (README_SCRIPT.md) FOR MORE INFORMATION ON THIS SCRIPT
#
#------------------------------------------------------------------------------------------------------------------------------------------

region = 'Amundsen'
where = 'j'

df_ua = pd.read_csv('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-05-22/Summary_UNN_Ua_2250.csv')

simulations = ['DC_ISSM','IGE_ElmerIce', 'ILTS_SICOPOLIS', 'LSCE_GRISLI2', 'NORCE_CISM2-MAR364-ERA-t1','PIK_PISM','UCM_Yelmo','ULB_fETISh-KoriBU2','UNN_Ua','UTAS_ElmerIce']
colors = ['darkblue', 'darkturquoise', 'darkorange', 'mediumaquamarine', 'limegreen', 'tomato', 'gold', 'olive', 'deeppink', 'mediumorchid']

path_ua = '/home/jovyan/private-storage/result/Flux/ligroundf_UNN_Ua_expAE04_2250.nc'
flux_target_ua = ismip.basin_flux_hand(xr.open_dataset(path_ua).ligroundf, region, where)
flux_target_ua = float(flux_target_ua.values)

# Combine all delta_flux and rmse_ua values across simulations
all_delta_flux = []
all_rmse_ua = []

fig = plt.figure(figsize=(15, 9))
for i, simu in enumerate(simulations):
    flux_ua = []
    df_simu = df_ua[df_ua['simulation'] == simu]
    exp = df_simu['experiment'].tolist()[0]
    year_simu = df_simu['year'].tolist()[0]

    path_rmse = f'/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-05-22/UNN_Ua_{simu}/RMSE_{exp}_2250.csv'
    df_rmse = pd.read_csv(path_rmse)
    rmse_values = df_rmse['rmse'].tolist()

    path = f'ligroundf_{simu}_{exp}.nc'
    flux_data = xr.open_dataset(path).ligroundf
    for year in range (len(flux_data.time)):
        flux_comp = ismip.basin_flux_hand(flux_data.isel(time=year), region, where)
        flux_ua.append(flux_comp)
    
    rmse_ua = np.array(rmse_values)
    flux_ua = np.array(flux_ua)

    delta_flux = np.abs(flux_ua - flux_target_ua)

    all_delta_flux.extend(delta_flux)
    all_rmse_ua.extend(rmse_ua)

    plt.scatter(delta_flux, rmse_ua, label=f'{simu} flux', marker='s', color=colors[i])

    # Perform linear regression for each simulation
    delta_flux = delta_flux.reshape(-1, 1)
    reg_simu = LinearRegression().fit(delta_flux, rmse_ua)
    slope_simu = reg_simu.coef_[0]
    intercept_simu = reg_simu.intercept_

    # Plot the regression line for each simulation
    x_vals_simu = np.linspace(min(delta_flux), max(delta_flux), 100).reshape(-1, 1)
    y_vals_simu = reg_simu.predict(x_vals_simu)
    plt.plot(x_vals_simu, y_vals_simu, color=colors[i], linestyle='--', label=f'{simu} Regression\n(slope={slope_simu:.4f})')

# Perform overall linear regression
all_delta_flux = np.array(all_delta_flux).reshape(-1, 1)
all_rmse_ua = np.array(all_rmse_ua)

reg = LinearRegression().fit(all_delta_flux, all_rmse_ua)
slope = reg.coef_[0]
intercept = reg.intercept_

# Plot the overall regression line
x_vals = np.linspace(min(all_delta_flux), max(all_delta_flux), 100).reshape(-1, 1)
y_vals = reg.predict(x_vals)

plt.plot(x_vals, y_vals, color='grey', linestyle='-', label=f'Overall Regression\n(slope={slope:.4f}, intercept={intercept:.4f})')
plt.xlabel(r'$\Delta F = |F_{comparison} - F_{target}|$', fontsize = 20)
plt.ylabel('RMSE', fontsize = 20)
plt.title('RMSE over relative ice flux at the grounding line in Amundsen', fontsize = 25, weight='bold')
plt.legend()
plt.savefig('RMSE_flux_reg.png', dpi=300)



