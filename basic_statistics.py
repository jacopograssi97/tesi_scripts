import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cpf
import netCDF4 as nc
from cdo import Cdo
cdo=Cdo()
from scipy import signal

def region_select(file_in, lon_min, lon_max, lat_min, lat_max, field, extract):
    
    """
    This function selects a region in a netdf file.
    It also allows to extract the data if needed
    """

    region = cdo.sellonlatbox(lon_min, lon_max, lat_min, lat_max, input=file_in)
    output = 0
    latitude = 0
    longitude = 0
    
    if extract == True:
        database = nc.Dataset(region)
        output = database[field][:]
        latitude = database['lat'][:]
        longitude = database['lon'][:]
        time = database['time'][:]
        database.close()

    return region, output, latitude, longitude



def histogram(variable_in, plot, title, x_label, save_name):
    
    """
    This function returns basics statistical indicator about the dataset
    It also allows to print a histogram
    """

    new_shp = 1
    for i in np.arange(0,variable_in.ndim,1):
        new_shp = new_shp*np.size(variable_in,i)
      
    var_res = variable_in.reshape(new_shp)

    variable_mean = np.mean(var_res)
    variable_std = np.std(var_res)
    
    if plot==True:

        fig = plt.figure(figsize=(8, 6), dpi=300, edgecolor='k')
        plt.hist((var_res))
        plt.savefig(save_name)

    return variable_mean, variable_std  

    
