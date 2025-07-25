import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
import config
import importlib
import sys

#load personal function
sys.path.append(f'{config.SAVE_PATH}/Function')
import Function.ISMIP_function as ismip

#=== Local Functions ===

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
def interp_field(field, x_gl, y_gl, method='linear'):
    if type(field) is xr.DataArray:
        interp = RegularGridInterpolator((y, x), field.values, bounds_error=False, fill_value=np.nan, method=method)
    else:
        interp = RegularGridInterpolator((y, x), field, bounds_error=False, fill_value=np.nan, method=method)
    return interp(np.column_stack((y_gl, x_gl)))

def check_neighbors(grounded_mask, x_gl, y_gl, resolution):
    """
    Check if any of the 8 neighboring points around each GL point are grounded.
    Returns a boolean array indicating if the GL point is close to a real grounded line.
    """
    close_to_real_gl = []
    for xi, yi in zip(x_gl, y_gl):
        # Check 8 directions at +-resolution
        check = (interp_field(grounded_mask, xi, yi, method='nearest') +
            interp_field(grounded_mask, xi + resolution, yi, method='nearest') +
            interp_field(grounded_mask, xi - resolution, yi, method='nearest') +
            interp_field(grounded_mask, xi, yi + resolution, method='nearest') +
            interp_field(grounded_mask, xi, yi - resolution, method='nearest'))
        if check < 5:
            close_to_real_gl.append(True)
        else:
            close_to_real_gl.append(False)
        
    return np.array(close_to_real_gl)


# === Parameters ===
grounding_line_level = 0.0  # Bed at sea level
density_ice = 917  # kg/m³
resample_spacing = 1000  # meters
max_velocity_threshold = np.inf  # m/yr (if velocity exceeds high values (/!\ it might affect the flux computation, set to np.Inf to ignore this step))

# === Load Data Using xarray ===
bed_data = xr.open_dataset(f'{config.SAVE_PATH}/Result/BedMachine.nc')
bed = ismip.grid_4x4(bed_data.bed)
bed = ismip.amundsen_mask(bed)

surface = xr.open_dataset(f'{config.SAVE_PATH}/Result/BedMachine.nc').surface
surface = ismip.grid_4x4(surface)
surface = ismip.amundsen_mask(surface)

thickness = xr.open_dataset(f'{config.SAVE_PATH}/Result/BedMachine.nc').thickness
thickness = ismip.grid_4x4(thickness)
thickness = ismip.amundsen_mask(thickness)

vitesse_vx = xr.open_dataset(f'{config.SAVE_PATH}/Result/antarctica_velocity_updated_v2.nc').vx
vitesse_vx = ismip.grid_4x4(vitesse_vx)
vx = ismip.amundsen_mask(vitesse_vx)

vitesse_vy = xr.open_dataset(f'{config.SAVE_PATH}/Result/antarctica_velocity_updated_v2.nc').vy
vitesse_vy = ismip.grid_4x4(vitesse_vy)
vy = ismip.amundsen_mask(vitesse_vy)

ice_base = (surface - thickness)
grounded_mask = np.abs(ice_base - bed).values < 1e-2  # Boolean grounded mask


flux = xr.zeros_like(bed)
flux.attrs.update({
    'units': 'Gt yr⁻¹',
    'long_name': 'Ice flux at the grounding line',
    'standard_name': 'grounding_line_flux'
})

# Crée un Dataset avec flux comme variable 'ligroundf'
flux_ds = xr.Dataset({'ligroundf': flux})

# Coordinates
x = bed["x"].values
y = bed["y"].values

# /!\ To adapt to the region of interest, you may need to adjust the coordinates or build a mask
# Masking the mesh except over a particular region (i.e. Amundsen Sea sector here)
# Masking: keep only region where x in [-2e6, -1e6] and y in [-1e6, 0]
x_mask = (x >= -2e6) & (x <= -1e6)
y_mask = (y >= -1e6) & (y <= 0)
region_mask = np.outer(y_mask, x_mask)
grounded_mask = np.where(region_mask, grounded_mask, 0)#, np.nan)

padded_mask = np.pad(grounded_mask, 1, mode='constant', constant_values=0)

# === Check if Contour Touches the Edge of the Array ===

# Find contours at the 0.5 level = boundary between grounded (1) and ungrounded (0)
contours = measure.find_contours(grounded_mask, level=0.5)

if not contours:
    raise RuntimeError("No contour found at level 0.5 — check your data.")
contour = max(contours, key=len)
contour_y, contour_x = contour.T  # in pixel coords

# Convert to real-world coordinates
dx = x[1] - x[0]
dy = y[1] - y[0]
x_real = x[0] + contour_x * dx
y_real = y[0] + contour_y * dy

x_gl, y_gl = resample_line(x_real, y_real, resample_spacing)

vx_gl = interp_field(vx, x_gl, y_gl)
vy_gl = interp_field(vy, x_gl, y_gl)
H_gl = interp_field(thickness, x_gl, y_gl)

# Check if the GL points are close to a real grounded line (on the mesh)
gl_check_status = check_neighbors(grounded_mask, x_gl, y_gl, dx)

# === Compute Local Normals ===
dx = np.gradient(x_gl)
dy = np.gradient(y_gl)
dl = np.sqrt(dx**2 + dy**2)
nx = dy / dl
ny = -dx / dl

# === Project Velocities onto Normals ===
vn = vx_gl * nx + vy_gl * ny
#vn =vn * 365.25 * 24 * 3600  # Convert to m/yr
vn[np.isnan(vn)] = 0
vn[vn > max_velocity_threshold] = 0.1

# === Compute Segmental Flux ===
flux = H_gl * vn * dl

# In case of masking of a region, mask boundaries should be removed from the flux computation 
# since they are not real grounded lines
flux[gl_check_status == False] = 0  # Set flux to 0 for points not close to a real grounded line 
flux[np.isnan(flux)] = 0  #remove possible NaN values 
total_flux_kg_per_yr = np.sum(flux) * density_ice
total_flux_Gt_per_yr = total_flux_kg_per_yr / 1e12

mass_flux = flux / dl / 1e6 # Gt/yr

mass_flux[mass_flux == 0] = np.nan  # Set zero flux to NaN for better visualization

for i, (x_line, y_line) in enumerate(zip(x_gl, y_gl)):
    selected_point = flux_ds.ligroundf.sel(x=x_line, y=y_line, method='nearest')
    x_idx, y_idx = selected_point.x.values, selected_point.y.values
    flux_ds.ligroundf.loc[dict(x=x_idx, y=y_idx)] += flux[i]

# === Save NetCDF ===
nc_path = f'{config.SAVE_PATH}/Result/ligroundf_bedmachine_test.nc'
flux_ds.to_netcdf(nc_path)

# === Output ===
print(f"Total integrated flux through grounding line: {total_flux_Gt_per_yr:.2f} Gt/yr")

# === Optional: Plot ===
plt.figure(figsize=(10, 6))
plt.title("Flux Along Grounding Line")
print((x[0], x[-1], y[0], y[-1]))
plt.imshow(grounded_mask, cmap="Greys", vmin = 0, vmax = 1 ,origin="lower", extent=(x[0], x[-1], y[0], y[-1]), alpha=0.5,zorder=-1)
plt.scatter(x_gl, y_gl, c=mass_flux,vmin = -0.05, vmax = 0.05, cmap="seismic", s=1)
plt.colorbar(label=r"Flux at the Grounding Line in (km$^3$/a per km)")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.tight_layout()
plt.savefig("GL_Flux_test2.png", dpi=300)