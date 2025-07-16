import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import directed_hausdorff
import xskillscore as xs
import config

### --------------- FUNCTIONS FOR ISMIP ANALYSIS --------------- ###
# --- FUNCTION SUMMARY ---
# choice()
# open_file(simu, experience, variable, where)
# plot_variable(simu, experience, variable, year, where)
# compute_grounding_mask(simu, experience, year, where)
# compute_grounding_mask_time(simu, experience, where)
# plot_grounded_mask(simu, experience, year, where)
# grid_interpolation(mask1, mask2)
# west_mask(mask, where)
# amundsen_mask(mask, where)
# amundsen_mask_flux(mask, where)
# get_resolution (data)
# amundsen_basin_flux(data, simu, where)
# compute_rmse(mask_target, mask, where)

def choice():
    """Help you choose the name of the simulation, experiment, variable and the year for the other function.

    Parameters
    -----------
    none

    Returns
    -------
    (string, string, string, string)
        simulation_name, experiment_name, variable_name, year

    Raises
    ------
    ValueError
        Error / Mistake in the name of any strings
    """
    ### ---------- NAME OF ALL THE SIMULATION AND VARIABLE AVAILABLE ---------- ###
    simulations = {
        'DC_ISSM' : '10 experiments + ctrl',
        'IGE_ElmerIce' : '6 experiments + ctrl + hist',
        'ILTS_SICOPOLIS' : '14 experiments + ctrl + hist',
        'IMAU_UFEMISM1' : '14 experiments + ctrl + historical',
        'IMAU_UFEMISM2' : '14 experiments + ctrl + historical',
        'IMAU_UFEMISM3' : '14 experiments + ctrl + historical',
        'IMAU_UFEMISM4' : '14 experiments + ctrl + historical',
        'LSCE_GRISLI' : '14 experiments + 2 ctrl + hist',
        'LSCE_GRISLI2' : '14 experiments + 2 ctrl + hist',
        'NORCE_CISM2-MAR364-ERA-t1' : '6 experiments + ctrl + historical',
        'NORCE_CISM3-MAR364-ERA-t1' : '6 experiments + ctrl + historical',
        'NORCE_CISM4-MAR364-ERA-t1' : '6 experiments + ctrl + historical',
        'NORCE_CISM4-MAR364-JRA-t1' : '6 experiments + ctrl + historical',
        'PIK_PISM' : '10 experiments + ctrl + historical',
        'UCM_Yelmo' : '14 experiments + ctrl + historical',
        'ULB_fETISh-KoriBU1' : '14 experiments + ctrl + historical',
        'ULB_fETISh-KoriBU2' : '14 experiments + ctrl + historical',
        'UNN_Ua' : '30 experiments (no 24) + ctrl + hist',
        'UTAS_ElmerIce' : '10 experiments + ctrl + hist',
        'VUB_AISMPALEO' : '10 experiments + ctrl + hist',
        'VUW_PRISM1' : '10 experiments (no 8 and 9) + ctrl + historical + init',
        'VUW_PRISM2' : '10 experiments (no 8 and 9) + ctrl',
    }
    
    variables = {
        "base": "base elevation",
        "topg": "bedrock elevation",
        "lithk": "ice thickness",
        "orog": "surface elevation",
        "xvelmean": "mean velocity in x",
        "yvelmean": "mean velocity in y",
        "strbasemag": "basal drag",
        "sftgif": "land ice area fraction",
        "sftgrf": "grounded ice sheet area fraction",
        "sftflf": "floating ice area fraction"
    }
    year_min = 2016
    year_max = 2301
    
    ### ---------- CHOISE OF YEAR, SIMULATION, EXPERIMENT, AND VARIABLE ---------- ###

    print('List of all the models:')
    for name in simulations.items():
        print(f" - {name[0]}")

    while True:
        simu_choice = input("\nEnter the model you want to use: ").strip()
        if simu_choice in simulations:
            break  # Sort de la boucle si l'entrée est valide
        print("Invalid model name. Please enter a valid model from the list.")
    
    simulations.get(simu_choice)
    experience_choice = input('\nEnter the simulation (experiment: expAE0X, control: ctrlAE, historical: historical) you want to plot')

    print("\nThe list of variables available is:")
    for key, description in variables.items():
        print(f" - {description} ({key})")
    
    while True:
        variable_choice = input('\nEnter the variable you want to plot:').strip()
        if variable_choice in variables:
            break
        print('Invalid variable name. Please enter a valid variable name from the list')
    

    while True:
        try:
            year_choice = int(input(f"\nEnter a year between {year_min} and {year_max}: ").strip())
            if year_min <= year_choice <= year_max:
                break
            else:
                print(f"The year must be between {year_min} and {year_max}. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a numeric year.")
    
    year = year_choice + '-01-01'

    print(f'\nYou chose the {simu_choice} model for the {experience_choice} run, and you want to plot the {variable_choice} variable in {year}')

    #SIMULATION WITH A DIFFERENTE NAME FOR HISTORICAL RUN
    simulations = ['IGE_ElmerIce', 'ILTS_SICOPOLIS', 'LSCE_GRISLI', 'LSCE_GRISLI2', 'UNN_Ua', 'UTAS_ElmerIce']

    #remove last letter of the experiment_choise if its historical
    if simu_choice in simulations:
        if experience_choice == 'historical':
            N = 6
            experience_choice = experience_choice[:-N]

    return simu_choice, experience_choice, variable_choice, year



def open_file(simu, experience, variable):
    """This function open the file corresponding to the simulation, experiment and variable you chose with the choice() function.

    Parameters
    ----------
        simulation : string, must be the mane of an ISMIP model
        experience : string, must be the name of an type of simulation; ctrl, expAEXX (where XX is a number; ex expAE04), historical
        variable : string, must be the name of variable 
        where : string, to signify where are you going to run the code, j; in juspyter, t; in the terminal

    Returns
    -------
    Dataset
        The Dataset you chose to open

    Raise
    -----
    Path Error
    """
 
    path_data = f'{config.DATA_PATH}/{simu}/{experience}/{variable}_AIS_{simu}_{experience}.nc'
    data = xr.open_dataset(path_data)
    
    return data


def plot_variable(simu, experience, variable, year):
    """This function plot the figure corresponding to your request of simulation, experiment, variable, and year.

    Parameters
    ----------
        simulation : string, must be the mane of an ISMIP model
        experience : string, must be the name of an type of simulation; ctrl, expAEXX (where XX is a number; ex expAE04), historical
        variable : string, must be tha name of variable
        year : string, year between 2016 and 2301, format example: 2046-01-01 (where only the year can be changed)

    Returns
    -------
    None
        figure that will be save in your current directory in png format.

    Raises
    ------
    """
    #ouverture des fichiers base et topo avec la fonction d'ouverture
    var = open_file(simu, experience, variable)
    
    #selection de l'année
    var_year = var[variable].sel(time=year)
    
    #plot de figure 
    plt.figure(figsize=(12, 9))
    var_year.plot()  
    plt.title(f'{simu} {experience} {variable} {year}', fontsize=18)
    plt.savefig(f'{simu}_{experience}_{variable}_{year}.png', dpi=300)

    print(f'Your figure is successfully plotted and save in you current directory as: {simu}_{experience}_{variable}_{year}.png')



def compute_grounding_mask(simu, experience, year):
    """This function compute the grounding mask for your chosen simulation, experiment and year.
    
    The function uses the base and topography of the model outputs and cumpute the diffrence between those two netCDF file.
    Three region will identify in the mask; 
        - 0: grounded ice sheet
        - 1: floatting ice shelf
        - 2: open ocean

    Parameters
    ----------
        simulation : string, must be the mane of an ISMIP model
        experience : string, must be the name of an type of simulation; ctrl, expAEXX (where XX is a number; ex expAE04), historical
        year : string, year between 2016 and 2301, format example: 2046-01-01 (where only the year can be changed)

    Returns
    -------
    Dataset(x, y)
        grounding mask of you chosing

    Raises
    ------
    
    """
    
    base_var = 'base'
    topo_var = 'topg'

    #### -------------- OUVERTURE DES FICHIERS -------------- ####
    base = open_file(simu, experience, base_var)
    topo = open_file(simu, experience, topo_var)

    #selection des variable et de l'année
    base_selec = base[base_var].sel(time=year)
    topo_selec = topo[topo_var].sel(time=year)


    #### -------------- DEFINITION DU MASK -------------- ####
    #mettre l'ocean en NaN
    base_selec_nan = base_selec.where(base_selec != 0, np.nan)
    topo_selec_nan = topo_selec.where(topo_selec != 0, np.nan)

    #calcul de la partie depossant sur le socle
    grounded = topo_selec_nan - base_selec_nan

    #passage des nan ocean en 2
    grounded_mask_nan = xr.where(~np.isnan(grounded), grounded, 2)

    #definition de la partie grounded
    cond1 = grounded_mask_nan < 0.01
    cond2 = grounded_mask_nan > -0.01
    condition = np.logical_and(cond1, cond2)
    grounded_mask = xr.where(condition, 0, grounded_mask_nan)
    
    #definition du mask
    grounded_condition = grounded_mask < 0
    grounded_mask_final = xr.where(grounded_condition, 1, grounded_mask)

    return grounded_mask_final

def compute_grounding_mask_time(simu, experience):
    """This function compute the grounding mask for your chosen simulation, experiment.
    
    The function uses the base and topography of the model outputs and cumpute the diffrence between those two netCDF file.
    Three region will identify in the mask; 
        - 0: grounded ice sheet
        - 1: floatting ice shelf
        - 2: open ocean

    Parameters
    ----------
        simulation : string, must be the mane of an ISMIP model
        experience : string, must be the name of an type of simulation; ctrl, expAEXX (where XX is a number; ex expAE04), historical

    Returns
    -------
    Dataset(x, y, time)
        grounding mask for all the time step

    Raises
    ------
    Path Error
    ValuesError
        size problem
    """
    
    base_var = 'base'
    topo_var = 'topg'

    #### -------------- OUVERTURE DES FICHIERS -------------- ####
    base = open_file(simu, experience, base_var)
    topo = open_file(simu, experience, topo_var)

    time = base.time.dt.year.values

    mask_list = []
    simulations = ['VUW_PISM1','VUW_PISM2']
    
    for year in time:
        if simu in simulations:
            date = str(year) + '-07-02'
        else:
            date = str(year) + '-01-01'
        base_selec = base[base_var].sel(time=date)
        topo_selec = topo[topo_var].sel(time=date)


    #### -------------- CREATION OF THE MASK -------------- ####
    #mask ocean values with NaN
        base_selec_nan = base_selec.where(base_selec != 0, np.nan)
        topo_selec_nan = topo_selec.where(topo_selec != 0, np.nan)

    #compute grounded part
        grounded = topo_selec_nan - base_selec_nan
        grounded_mask_nan = xr.where(~np.isnan(grounded), grounded, 2)#raise issue if you don't run this line

        cond1 = grounded_mask_nan < 0.01
        cond2 = grounded_mask_nan > -0.01
        condition = np.logical_and(cond1, cond2)
        grounded_mask = xr.where(condition, 0, grounded_mask_nan)
    
    #define mask
        grounded_condition = grounded_mask < 0
        grounded_mask_final = xr.where(grounded_condition, 1, grounded_mask)

        #mask_list.append(grounded_mask_final.assign_coords(time=year))
        mask_list.append(grounded_mask_final)
        print(f"Creation of the mask: {year}")


    grounding_mask_ds = xr.concat(mask_list, dim='time')
    grounded_dataset = xr.Dataset({"grounding_mask" : grounding_mask_ds})
    #grounded_dataset.to_netcdf(f"grounding_mask_{simu}_{experience}.nc")
    print(f"netCDF file {simu} {experience} successfully created")
    return grounded_dataset

def plot_grounded_mask(simu, experience, year):
    """This function plot the grounding mask for the simulation, experiment and year you specfied

    Parameters
    ----------
        simulation : string, must be the mane of an ISMIP model
        experience : string, must be the name of an type of simulation; ctrl, expAEXX (where XX is a number; ex expAE04), historical
        year : string, year between 2016 and 2301, format example: 2046-01-01 (where only the year can be changed)

    Return
    ------
    None
        figure that will be save in your current directory in png format.
    
    """
    grounded_mask = compute_grounding_mask(simu, experience, year)
    
    #### -------------- PLOT -------------- ####
    fig, ax = plt.subplots(figsize=(12, 10))
    #color definition
    color = ListedColormap(['wheat', 'cadetblue', 'aliceblue'])
    
    ax.pcolormesh(grounded_mask.x, grounded_mask.y, grounded_mask[0], cmap=color, vmin=0, vmax=2)

    #definition des legendes des couleurs du mask
    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='wheat', markersize=10, label='Grounded (0)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='cadetblue', markersize=10, label='Floating (1)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='aliceblue', markersize=10, label='Ocean (2)')
    ]
    ax.legend(handles=handles, loc='upper right')

    #
    ax.set_ylabel('y-coordinate in Cartesian system [m]')
    ax.set_xlabel('y-coordinate in Cartesian system [m]')
    plt.title(f'Grounded mask for {year}: {simu} {experience}', fontsize=18)
    
    plt.savefig(f'Grounded_mask_{simu}_{experience}_{year}.png', dpi=300)

    print(f'You grounded mask is successfully plotted and save in you current directory as: Grounded_mask_{simu}_{experience}_{year}.png')



def grid_interpolation(mask1, mask2):
    """This function interpolate grid point if mask1 and mask2 aren't the same size

    Parameters
    ----------
        mask1 : dataset, please use the compute_grounding_mask before on the dataset
        mask2 : dataset, please use the compute_grounding_mask before on the dataset

    Returns
    -------
    (Dataset, Dataset)
        The 2 dataset interpolated if needded
    """
    if mask1.x.shape != mask2.x.shape:
        if mask1.x.shape > mask2.x.shape:
            mask2 = mask2.interp(x = mask1.x, y = mask1.y)
            #print('mask2 has been interpolated')
        else:
            mask1 = mask1.interp(x = mask2.x, y = mask2.y)
            #print('mask1 has been interpolated')
    return mask1, mask2


def amundsen_mask(mask):
    """Select the Amundsen region using the Zwally regional definition

    Parameters
    ----------
        mask : dataset, the mask onto you want to select the Amundsen regions
        where : string, where de you run the code ('j' in Jupyter Notebook, other in the terminal)

    Returns
    -------
    (Dataset)
        Dataset with Amundsen regions mask
    """
    # chemin du mask en fonction d'ou on run le code
    mask_ant = xr.open_dataset(f'{config.SAVE_PATH}/Result/Basins_Zwally_8km.nc')
    
    # Interpolation sur la grille du mask (Elmer ou autre)
    mask_ant_int = grid_4x4(mask_ant)
    
    region = mask_ant_int.Basin_ID
    region_amundsen = [21, 22]
    mask_bool = np.isin(region, region_amundsen)
    
    mask_amundsen = mask.where(mask_bool)

    return mask_amundsen


def get_resolution(data):
    """Give the resolution of a netCDF grid in meters

    Parameters
    ----------
        data : dataset, (e.g ice flux at the grounding line)

    Returns
    -------
    (float)
        Grid resolution
    """
    x = data.x
    x0 = x.isel(x = 0)
    x1 = x.isel(x = 1)

    reso = abs(x1 - x0)
    return reso.values.item()

def basin_flux(data, simu, region):
    """Compute Ice flux at in grounding line for Amundsen basin

    Parameters
    ----------
        data : dataset, ice flux at the grounding line
        simu : string, simulation on which you do the computation
        where : string, where de you run the code ('j' in Jupyter Notebook, other in the terminal)

    Returns
    -------
    (float)
        Ice flux for the Amundsen basin
    """
    mask_ant = xr.open_dataset(f'{config.SAVE_PATH}/Result/Basins_Zwally_8km.nc')
    mask_ant_interp = mask_ant.interp(x = data.x, y = data.y)
    
    reso = get_resolution(data)#get grid resolution in m
    cell_area = reso*reso

    #selection of region
    if region == 'Ross':
        region = mask_ant_interp.Basin_ID
        region_ross = [18, 19]
        mask_bool = np.isin(region, region_ross)
    
        data_region = data.where(mask_bool)
    else:
        region = mask_ant_interp.Basin_ID
        region_amundsen = [21, 22]
        mask_bool = np.isin(region, region_amundsen)
    
        data_region = data.where(mask_bool)

    fetish = ['ULB_fETISh-KoriBU2']
    unn = ['UNN_Ua']
    
    if simu in fetish:
        data_region = data_region / 16000**2
    if simu in unn:
        data_region = data_region / 10

    convert = (365.25*24*60*60*cell_area)/1e12 #convsertion kg/s.m^2 in Gt/yr
    #flux_amun = np.nansum(abs(data_amun))*convert
    flux = abs(data_region*convert).sum(dim = ['x', 'y'], skipna = True)
    return flux

def basin_flux_hand(data, region):
    """Compute Ice flux at in grounding line for Amundsen basin

    Parameters
    ----------
        data : dataset, ice flux at the grounding line
        simu : string, simulation on which you do the computation
        where : string, where de you run the code ('j' in Jupyter Notebook, other in the terminal)

    Returns
    -------
    (float)
        Ice flux for the Amundsen basin
    """
    mask_ant = xr.open_dataset(f'{config.SAVE_PATH}/Result/Basins_Zwally_8km.nc')
    mask_ant_interp = mask_ant.interp(x = data.x, y = data.y)
    density_ice = 917

    #selection of region
    if region == 'Ross':
        region = mask_ant_interp.Basin_ID
        region_ross = [18, 19]
        mask_bool = np.isin(region, region_ross)
    
        data_region = data.where(mask_bool)
    else:
        region = mask_ant_interp.Basin_ID
        region_amundsen = [21, 22]
        mask_bool = np.isin(region, region_amundsen)
    
        data_region = data.where(mask_bool)

    flux = (abs(data_region).sum(dim = ['x', 'y'], skipna = True) * density_ice)/1e12
    return flux

def grid_4x4(mask):
    ref_grid_data = xr.open_dataset(f'{config.SAVE_PATH}/Result/grid4x4.nc')
    ref_grid = ref_grid_data.grounding_mask

    mask_interp = mask.interp(x = ref_grid.x, y = ref_grid.y)

    return mask_interp

def compute_rmse(mask_target, mask, region):
    """Comute the RMSE between a target Dataset and the experiment of another simulation

    Parameters
    ----------
        mask_target : Dataset, target grounding mask at a given time
        mask : Dataset, masks you want to compare
        region : string, 'Ross' or 'Amundsen'
        where : string, where de you run the code ('j' in Jupyter Notebook, other in the terminal)

    Returns
    -------
    (array)
        RMSE of each time step between the target mask and comparaison masks
    
    """
    time_dim = mask.time.shape
    date = time_dim[0]
    rmse = np.full(date, np.nan)

    mask_target_interp = grid_4x4(mask_target)

    for t in range(date):
        mask_t = mask.grounding_mask.isel(time=t)
        mask_t_interp = grid_4x4(mask_t)
          
        if region == 'Amundsen':
            mask_target_region = amundsen_mask(mask_target_interp)
            mask_t_region = amundsen_mask(mask_t_interp)

        condition_rmse_target = (mask_target_region==0)
        condition_rmse_t = (mask_t_region==0)

        #0: grounded, 1: else
        mask_target_rmse = xr.where(condition_rmse_target, mask_target_region, 1)
        mask_t_rmse = xr.where(condition_rmse_t, mask_t_region, 1)

        rmse_value = xs.rmse(mask_target_rmse, mask_t_rmse)
        rmse[t] = rmse_value.values.item()
    
    return np.array(rmse)

    