import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cpf
import netCDF4 as nc
from cdo import Cdo
cdo=Cdo()
from scipy import signal
from basic_statistics import *

file_in = '~/work/jacopo/DATA/ERA5_prec_day_HKKH.nc'

dummy, orog_HKK, latitude_HKK, longitude_HKK = region_select(file_in, 71, 78, 32, 37, 'tp', True)

mean, std = histogram(orog_HKK, True, 'Istogramma orografia', 'quota [m]','prova.png')

print(mean)