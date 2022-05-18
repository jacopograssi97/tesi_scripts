import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cpf
import netCDF4 as nc
from cdo import Cdo
from scipy.fftpack import fft
cdo=Cdo()
from scipy import signal
from datetime import date, timedelta

def extract_nc(file_in, field, extract_time, units_time, calendar):
    
    """
    This function extracts the data from a .nc file
    """

    database = nc.Dataset(file_in, 'r')
    output = database[field][:]
    time = database['time'][:]

    # ERA5 is incoherent in his orography files...
    if field == 'z':
        latitude = database['latitude'][:]
        longitude = database['longitude'][:]
    else:
        latitude = database['lat'][:]
        longitude = database['lon'][:]

    database.close()

    if extract_time == True:
        time = nc.num2date(time, units_time, calendar)

    return output, latitude, longitude, time



def altitude_mask(field, orography, bottom_level, top_level):
    
    """
    This function returns basics statistical indicator about the dataset
    It also allows to print a histogram
    """
    masked_field = np.ma.masked_where(orography<bottom_level,field)
    masked_field = np.ma.masked_where(orography>top_level,masked_field)
    
    return masked_field

    
    

    
