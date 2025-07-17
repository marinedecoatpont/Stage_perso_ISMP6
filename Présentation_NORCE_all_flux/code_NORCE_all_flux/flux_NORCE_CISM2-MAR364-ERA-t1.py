import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
import importlib
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
import config
import os

#LOADING PERSONAL FUNCTION (LIBRARY)
sys.path.append(f'{config.SAVE_PATH}/Function')
import Function.ISMIP_function as ismip
importlib.reload(ismip)

# LOCAL FUNCTION
# === Resample Grounding Line ===
def resample_line(x, y, spacing):
    """Compute the real distance between the coordinate vector

    Input:
    ------
        - x: np.array, vecor coordinnate
        - y: np.array, vector coordinnate
        - spacing: float, distance between two coordinnate point

    Returns:
    --------
        - x_new: np.array, new coordinnate 
        - y_new, np.array, new coordinate
    """
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    dist_cum = np.concatenate([[0], np.cumsum(dists)])
    n_points = int(dist_cum[-1] // spacing)
    new_distances = np.linspace(0, dist_cum[-1], n_points)
    x_new = np.interp(new_distances, dist_cum, x)
    y_new = np.interp(new_distances, dist_cum, y)
    return x_new, y_new


# === Interpolate Fields at GL Points ===
def interp_field(field):
    """Interpolates the values of a given field at the grounding line points.

    This function uses a regular grid interpolator to compute the values of 
    the input field at specified grounding line coordinates.

    Input:
    ------
        - field : xarray.DataArray, The 2D field to be interpolated. It is assumed to have dimensions corresponding to the y and x coordinates.

    Returns:
    --------
        - numpy.ndarray, An array containing the interpolated values of the field at the grounding line points.

    Notes:
    ------
    - The variables `y`, `x`, `y_gl`, and `x_gl` must be defined in the 
      global scope or passed to the function for it to work correctly.
    - The interpolation will return NaN for points outside the bounds of 
      the input field.
    """
    interp = RegularGridInterpolator((y, x), field.values, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack((y_gl, x_gl)))


# === Parameters ===
grounding_line_level = 0.0  # Bed at sea level
density_ice = 917  # kg/m³
resample_spacing = 1000  # meters
max_velocity_threshold = 2000  # m/yr

condition = ['ULB_fETISh-KoriBU2', 'UNN_Ua']

# CSV file for the different scenarios
dfs = '/Users/marine/Documents/Master/M2/Stage/code_openreprolab/Result/min_RMSE_UNN_Ua_2250_NORCE_CISM2-MAR364-ERA-t1.csv'

#opening of dataset of bed machine as a template for the netCDF ouput
data_bedmachine = xr.open_dataset(f'{config.SAVE_PATH}/Result/ligroundf_bedmachine.nc')

df = pd.read_csv(dfs)

for index, row in df.iterrows():
    simu = row['simulation']
    exp = row['experiment']
    year_simu = int(row['min_year'])
    year_index = year_simu - 2016

    #creat a list 'time' to compute ±5 years before and after the RMSE minimum
    times = [year_index]
    for i in range(1, 6):
        if (year_simu - i) >= 2017:
            times.append(year_index - i)

    for i in range(1, 6):
        if (year_simu + i) <= 2299:
                times.append(year_index + i)
    print(times)  

    #start of the ice flux conputation
    for time in times:
        # Check if the file already exists
        output_file = f'{config.PATH_IF}/{simu}/ligroundf_{simu}_{exp}_{time+2016}.nc'
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping computation.")
            continue

        #file opening
        ds_orog = ismip.open_file(simu, exp, 'orog')
        surface = ds_orog.orog.isel(time = time)
    
        ds_topg = ismip.open_file(simu, exp, 'topg')
        bed = ds_topg.topg.isel(time = time)
    
        ds_thk = ismip.open_file(simu, exp, 'lithk')
        thickness = ds_thk.lithk.isel(time = time)

        #condition for some models which do not have right time in dataset
        if simu in condition:
            ds_vx = xr.open_dataset(f'{config.DATA_PATH}/{simu}/{exp}/xvelmean_AIS_{simu}_{exp}.nc', decode_times = False)
            ds_vy = xr.open_dataset(f'{config.DATA_PATH}/{simu}/{exp}/yvelmean_AIS_{simu}_{exp}.nc', decode_times = False)
        else:
            ds_vx = ismip.open_file(simu, exp, 'xvelmean')
            ds_vy = ismip.open_file(simu, exp, 'yvelmean')
        vx = ds_vx.xvelmean.isel(time = time)
        vy = ds_vy.yvelmean.isel(time = time)

        #creation for a blank netCDF file for ice flux output
        flux_ds = xr.zeros_like(data_bedmachine)
        flux_ds = flux_ds.interp(x = ds_orog.x, y = ds_orog.y)


        ice_base = (surface - thickness)
        grounded_mask = np.abs(ice_base - bed).values < 1e-2  # Boolean grounded mask
        gl_mask = grounded_mask.astype(float)

        x = bed["x"].values
        y = bed["y"].values

        contours = measure.find_contours(grounded_mask, level=0.5)
        contour = max(contours, key=len)
        contour_y, contour_x = contour.T  # in pixel coords

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        x_real = x[0] + contour_x * dx
        y_real = y[0] + contour_y * dy

        x_gl, y_gl = resample_line(x_real, y_real, resample_spacing)
        vx_gl = interp_field(vx)
        vy_gl = interp_field(vy)
        H_gl = interp_field(thickness)

        # Compute normals
        dx = np.gradient(x_gl)
        dy = np.gradient(y_gl)
        dl = np.sqrt(dx**2 + dy**2)
        nx = dy / dl
        ny = -dx / dl

        vn = vx_gl * nx + vy_gl * ny
        #ajouter condition pour les calcul pour bedmachine car les vitesse sont deja a la bonne unite
        vn =vn * 365.25 * 24 * 3600  # Convert to m/yr
        vn[np.isnan(vn)] = 0
        vn[vn > max_velocity_threshold] = 0.1

        # Compute Segmental Flux
        flux = H_gl * vn * dl #m3/yr

        #save in netCDF
        print('converte to netCDF')
        for i, (x_line, y_line) in enumerate(zip(x_gl, y_gl)):
            selected_point = flux_ds.ligroundf.sel(x=x_line, y=y_line, method='nearest')

            x_idx, y_idx = selected_point.x.values, selected_point.y.values

            flux_ds.ligroundf.loc[dict(x=x_idx, y=y_idx)] += flux[i]

        # === Save NetCDF ===
        path = f'{config.PATH_IF}/{simu}'
        os.makedirs(path, exist_ok=True)
        flux_ds.to_netcdf(output_file)