import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cdo import Cdo
cdo=Cdo()
from basic_op import *

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

# Selecting netCDF files 
file_in_prec = '~/work/jacopo/DATA/ERA5_prec_day_HKKH.nc'
file_in_oro = '~/work/jacopo/DATA/ERA5_orog_HKKH.nc'

# Selecting region [comment unwanted]
# Hindu-Kush Karakoram
#region_prec = cdo.sellonlatbox(71, 78, 32, 37, input = file_in_prec)
#region_oro = cdo.sellonlatbox(71, 78, 32, 37, input = file_in_oro)

# Himalaya
#region_prec = cdo.sellonlatbox(78, 93, 25, 32, input = file_in_prec)
#region_oro = cdo.sellonlatbox(78, 93, 25, 32, input = file_in_oro)

# All
region_prec = cdo.sellonlatbox(70,95,20,40, input = file_in_prec)
region_oro = cdo.sellonlatbox(70,95,20,40, input = file_in_oro)

# Extracting data and operating transormation
prec, latitude, longitude, time = extract_nc(region_prec, 'tp', True, 'hours since 1950-01-01', 'proleptic_gregorian')
oro, latitude, longitude, dummy = extract_nc(region_oro, 'z', False, 0, 0)
oro = np.squeeze(oro/9.81)
prec = prec*1000*24

# Creating profile based on altitude
step = 5  # step for discretizing altitude
mean_map = np.squeeze(np.mean(prec, 0))
prof_mean = []

for i in np.arange(0,9000,step):

    masked_rain = altitude_mask(mean_map, oro, i, i+100)
    prof_mean.append(np.squeeze(np.mean(masked_rain)))


fig = plt.figure(figsize=(8, 6), dpi=300, edgecolor='k')
plt.plot(np.arange(0,9,step/1000),prof_mean)
plt.show()
plt.grid()
plt.xlabel('Altitude [km]')
plt.ylabel('Tp [mm/day]')
plt.ylim((0,14))
plt.savefig('Grafici/profilo_all.png')