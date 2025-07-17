import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
import importlib
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
import os

#load personal function
sys.path.append('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/Fonction')
import ISMIP_function as ismip
importlib.reload(ismip)

#---------------------------------- COMPUTATION OF ICE FLUX AT THE GROUNDING LINE  -----------------------------------------------
#AUTHOR: CYRILLE MOSBEUX
#MODIFICATION: marine de coatpont
#April 18, 2025
#IGE / ISMIP6 internship
#
#
#PLEASE READ README () FOR MORE INFORMATION ON THIS SCRIPT
#
#------------------------------------------------------------------------------------------------------------------------------------------

# === Resample Grounding Line ===
def resample_line(x, y, spacing):
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    dist_cum = np.concatenate([[0], np.cumsum(dists)])
    n_points = int(dist_cum[-1] // spacing)
    new_distances = np.linspace(0, dist_cum[-1], n_points)
    x_new = np.interp(new_distances, dist_cum, x)
    y_new = np.interp(new_distances, dist_cum, y)
    return x_new, y_new


# === Interpolate Fields at GL Points ===
def interp_field(field):
    interp = RegularGridInterpolator((y, x), field.values, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack((y_gl, x_gl)))


print('----- BEGINING OF PLOT PROGRAM -----')

# === Parameters ===
grounding_line_level = 0.0  # Bed at sea level
density_ice = 917  # kg/m³
resample_spacing = 1000  # meters
max_velocity_threshold = 2000  # m/yr
where = 'j'

df = pd.read_csv('/home/jovyan/private-storage/Decoatpont_m2_ISMP6_personal/2025-07-17/Summary_ULB_fETISh-KoriBU1_2025.csv')
simulations = ['ULB_fETISh-KoriBU2', 'ULB_fETISh-KoriBU1']
condition = ['ULB_fETISh-KoriBU2', 'ULB_fETISh-KoriBU1']

for index, row in df.iterrows():
    simu = row['simulation']
    exp = row['experiment']
    year_simu = int(row['min_year'])
    year_index = year_simu - 2016
        
    #file for hand computation
    ds_orog = ismip.open_file(simu, exp, 'orog', where)
    ds_topg = ismip.open_file(simu, exp, 'topg', where)
    ds_thk = ismip.open_file(simu, exp, 'lithk', where)
        
    if simu in condition:
        ds_vx = xr.open_dataset(f'/home/jovyan/private-storage/simu/{simu}/{exp}/xvelmean_AIS_{simu}_{exp}.nc', decode_times = False)
        ds_vy = xr.open_dataset(f'/home/jovyan/private-storage/simu/{simu}/{exp}/yvelmean_AIS_{simu}_{exp}.nc', decode_times = False)
    else:
        ds_vx = ismip.open_file(simu, exp, 'xvelmean', where)
        ds_vy = ismip.open_file(simu, exp, 'yvelmean', where)
        

    # === Initialize Output Dataset ===
    flux_ds = xr.zeros_like(ds_orog).rename({'orog': 'ligroundf'})
    flux_ds['ligroundf'].attrs.update({
            'units': 'Gt yr⁻¹',
            'long_name': 'Ice flux at the grounding line',
            'standard_name': 'grounding_line_flux'
        })
    flux_ds.attrs.update({
            'title': 'Grounding line ice flux time series',
            'summary': 'Mass fluxes computed along the Antarctic grounding line for each year',
            'source': 'ISMIP6 model outputs',
            'history': 'Created on 2025-05-14 using custom script',
            'comment': 'Computed using custom script and ISMIP6 outputs',
            'Conventions': 'CF-1.8',
            'note': 'Resampled to 1 km resolution along grounding line'
        })
    filepath = f'Flux/ligroundf_{simu}_{exp}.nc'
    
    if os.path.exists(filepath):
        print(f'Le fichier {filepath} existe déjà, on passe à la suite.')
        continue
        
    for time in range (len(ds_orog.time)):
        print(f'{exp}: Year {time + 1}/{len(ds_orog.time)}')
            
        # === Homemade computation ===
        surface = ds_orog["orog"].isel(time=time)
        surface = ismip.grid_4x4(surface)
        surface = ismip.amundsen_mask(surface, where)
        
        bed = ds_topg["topg"].isel(time=time)
        bed = ismip.grid_4x4(bed)
        bed = ismip.amundsen_mask(bed, where)
        
        thickness = ds_thk["lithk"].isel(time=time)
        thickness = ismip.grid_4x4(thickness)
        thickness = ismip.amundsen_mask(thickness, where)
        
        vx = ds_vx["xvelmean"].isel(time=time)
        vx = ismip.grid_4x4(vx)
        vx = ismip.amundsen_mask(vx, where)
        
        vy = ds_vy["yvelmean"].isel(time=time)
        vy = ismip.grid_4x4(vy)
        vy = ismip.amundsen_mask(vy, where)
        
        ice_base = (surface - thickness)

        grounded_mask = np.abs(ice_base - bed).values < 1e-2  # Boolean grounded mask
        gl_mask = grounded_mask.astype(float)

        x = bed["x"].values
        y = bed["y"].values

        # Find contours at the 0.5 level = boundary between grounded (1) and ungrounded (0)

        contours = measure.find_contours(grounded_mask, level=0.5)

        if not contours:
            raise RuntimeError("No contour found at level 0.5 — check your data.")


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
        vn =vn * 365.25 * 24 * 3600  # Convert to m/yr
        vn[np.isnan(vn)] = 0
        vn[vn > max_velocity_threshold] = 0.1

        # Compute Segmental Flux
        flux = H_gl * vn * dl#m3/yr
        flux[np.isnan(flux)] = 0
        total_flux_kg_per_yr = np.sum(flux) * density_ice
        total_flux_Gt_per_yr = total_flux_kg_per_yr / 1e12
        mass_flux = flux / dl / 1e6 # Gt/yr

        #loop on coordinates in order to have a dataset
        print('converting to dataset')

        for i, (x_line, y_line) in enumerate(zip(x_gl, y_gl)):
            selected_point = flux_ds.ligroundf.isel(time = time).sel(x=x_line, y=y_line, method='nearest')

            x_idx, y_idx = selected_point.x.values, selected_point.y.values

            flux_ds.ligroundf.isel(time = time).loc[dict(x=x_idx, y=y_idx)] += flux[i]

    # === Save NetCDF ===
    nc_path = f'ligroundf_{simu}_{exp}.nc'
    flux_ds.to_netcdf(nc_path)
    print('netCDF saved!')



