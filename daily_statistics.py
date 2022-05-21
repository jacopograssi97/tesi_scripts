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

# Selecting region [comment unwanted]
# Hindu-Kush Karakoram
#region_prec = cdo.sellonlatbox(71, 78, 32, 37, input = file_in_prec)

# Himalaya
#region_prec = cdo.sellonlatbox(78, 93, 25, 32, input = file_in_prec)

# All
region_prec = cdo.sellonlatbox(70,95,20,40, input = file_in_prec)

# Extracting time serie (mean on space)
time_serie = cdo.fldmean(input=region_prec)

# Performing mean and std for each month
avg_month = cdo.ydaymean(input=time_serie)
min_month = cdo.ydaymin(input=time_serie)
max_month = cdo.ydaymax(input=time_serie)


# Extracting data and operating transormation
avg_month, latitude, longitude, time = extract_nc(avg_month, 'tp', True, 'hours since 1950-01-01', 'proleptic_gregorian')
min_month, dummy, dummy, dummy = extract_nc(min_month, 'tp', False, 0, 0)
max_month, dummy, dummy, dummy = extract_nc(max_month, 'tp', False, 0, 0)

avg_month = np.squeeze(avg_month*1000*24)
min_month = np.squeeze(min_month*1000*24)
max_month = np.squeeze(max_month*1000*24)


fig = plt.figure(figsize=(8, 6), dpi=300, edgecolor='k')
plt.plot(np.arange(0,366,1), avg_month, label='avg')
plt.plot(np.arange(0,366,1), min_month, label='min')
plt.plot(np.arange(0,366,1), max_month, label='max')
plt.legend()
plt.xlabel('Gregorian day')
plt.ylabel('Tp [mm/day]')
plt.grid()
plt.show()
plt.savefig('Grafici/avg_day_all.png')