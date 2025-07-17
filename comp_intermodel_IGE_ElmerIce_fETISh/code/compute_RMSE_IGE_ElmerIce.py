import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xskillscore as xs
import sys
import importlib
import pandas as pd
import os

#load personal function
sys.path.append('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/Fonction')
import ISMIP_function as ismip
importlib.reload(ismip)

#----------------------- COMPUTE RMSE FOR A TARGET MASK AT A GIVEN YEAR AND EXPERIMENT OF ANOTHER SIMULATION -------------------------------
#marine de coatpont
#April 18, 2025
#IGE / ISMIP6 internship
#
#
#PLEASE READ README () FOR MORE INFORMATION ON THIS SCRIPT
#
#------------------------------------------------------------------------------------------------------------------------------------------

#target and comparison choice
target_simu = 'IGE_ElmerIce'
target_exp = 'expAE05'
target_year = '2273'
comp_simu = 'IGE_ElmerIce'
where = 'j'
region = 'Amundsen'

target_index = int(target_year) - 2016
target_data = xr.open_dataset(f'/home/jovyan/private-storage/result/Grounding_mask/{target_simu}/grounding_mask_{target_simu}_{target_exp}.nc')
target_mask = target_data.grounding_mask.isel(time = target_index)

#comparaison mask parameters (change)
simulation = {
    'DC_ISSM' : 10 ,
    'IGE_ElmerIce' : 6,
    'ILTS_SICOPOLIS' : 14,
    'LSCE_GRISLI2' : 14,
    'NORCE_CISM2-MAR364-ERA-t1' : 6,
    'PIK_PISM' : 10,
    'UCM_Yelmo' : 14,
    'ULB_fETISh-KoriBU2' : 14,
    'UNN_Ua' : 30,
    'UTAS_ElmerIce' : 10,
    'VUB_AISMPALEO' : 10,
    'VUW_PRISM1' : 10,
    'VUW_PRISM2' : 10,
}
vuw = ['VUW_PRISM1', 'VUW_PRISM2'] #for a following condition

min_table = []
summary_table = []
min_table_simu = []


nb_exp = simulation[comp_simu]
exp_list = [f'expAE{str(i).zfill(2)}' for i in range(1, nb_exp + 1)]

#some simulation don't have all the experiment
if comp_simu == 'UNN_Ua':
    exp_list = [exp for exp in exp_list if exp not in ['expAE24']]
if comp_simu in vuw:
    exp_list = [exp for exp in exp_list if exp not in ['expAE08', 'expAE09']]

colors = plt.get_cmap('tab20').colors
plt.figure(figsize=(12, 6))   
 
for i, exp in enumerate(exp_list):
    print(f'Experiement {i+1}/{len(exp_list)}')
    comp_data = xr.open_dataset(f'/home/jovyan/private-storage/result/Grounding_mask/{comp_simu}/grounding_mask_{comp_simu}_{exp}.nc')

    rmse_target_comp = ismip.compute_rmse(target_mask, comp_data, region, where)

    time = comp_data.time.values
    years = comp_data.time.dt.year.values

    min_rmse = np.nanmin(rmse_target_comp)
    min_year = years[np.nanargmin(rmse_target_comp)]
    print(f'Minimale RMSE for {exp} is: {min_rmse:.4f} in {min_year}')

    #add to min_table
    min_table.append({'experiment': exp, 'min_RMSE': min_rmse, 'min_year': min_year})

    #plot RMSE during time
    color = colors[i % len(colors)]

    plt.plot(years, rmse_target_comp, color=color, label=f'{comp_simu} {exp} (min={min_rmse:.4f} in {min_year})')

    #save RSME in csv
    df = pd.DataFrame({
            "time": time,
            "year": years,
            "rmse" : rmse_target_comp
    })
    save_dir = f'{target_simu}_{comp_simu}_{target_year}'
    os.makedirs(save_dir, exist_ok=True)

    df.to_csv(f'{save_dir}/RMSE_{exp}.csv', index=False)
    print(f'CSV file {exp} saved successfully!')


#plot
plt.title(f'RMSE of {target_simu} {target_exp} and {comp_simu} experiments', fontsize=18)
plt.xlabel('Year')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{target_simu}_{comp_simu}_{target_year}/RMSE_{target_simu}_{target_year}_{comp_simu}.png', dpi=300)
print('Figure saved successfully')

#min table
min_df = pd.DataFrame(min_table)
min_df.to_csv(f'{target_simu}_{comp_simu}_{target_year}/min_RMSE_{target_simu}_{target_year}_{comp_simu}.csv', index=False)
print("Minimum table saved successfully")

#global summary CSV 
mini = pd.read_csv(f'{target_simu}_{comp_simu}_{target_year}/min_RMSE_{target_simu}_{comp_simu}.csv')
rmse = mini['min_RMSE'].values
date = mini['min_year'].values
experiment = mini['experiment'].values
mini_rmse = np.nanmin(rmse)
mini_exp = experiment[np.nanargmin(rmse)]
mini_year = date[np.nanargmin(rmse)]

summary_table.append({'simulation': comp_simu, 'experiment': mini_exp, 'year': mini_year, 'RMSE': mini_rmse})
print(f'All done for {comp_simu}')

summary_df = pd.DataFrame(summary_table)
summary_df.to_csv(f'Summary_{target_simu}_{comp_simu}.csv', index=False)
print(f'CSV summary file {target_simu} {comp_simu} saved successfully')

print('Everything seems fine ~(°w°~)')
print('End of the script')