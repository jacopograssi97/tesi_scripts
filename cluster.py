import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import cartopy.feature as cpf
from cdo import Cdo
cdo=Cdo()
from basic_op import *
import bottleneck as bn

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

# Selecting netCDF files 
file_in_prec = '~/work/jacopo/DATA/ERA5_prec_day_HKKH.nc'

# Selecting region [comment unwanted]
# Hindu-Kush Karakoram
region_prec = cdo.sellonlatbox(71, 78, 32, 37, input = file_in_prec)

# Himalaya
#region_prec = cdo.sellonlatbox(78, 93, 25, 32, input = file_in_prec)

# All
#region_prec = cdo.sellonlatbox(71,93,25,37, input = file_in_prec)

# Extracting time serie (mean on space)
avg_month = cdo.ydaymean(input=region_prec)

# Extracting data and operating transormation
prec_fields, latitude, longitude, time = extract_nc(avg_month, 'tp', True, 'hours since 1950-01-01', 'proleptic_gregorian')
prec_fields = np.squeeze(prec_fields*1000*24)

eof, pc = eof_custom(prec_fields, 2, 2, True)
to_fit = np.squeeze(pc)

to_fit = (to_fit - np.mean(to_fit, axis=0))/np.std(to_fit,axis=0)

clust, clust_centers = kmean_custom(to_fit, 4, 300, True)

fig = plt.figure(figsize=(8, 6), dpi=300, edgecolor='k')
plt.scatter(np.arange(0,366,1), to_fit[:,0], c=clust, cmap='jet' )
plt.xlabel('Gregorian Day')
plt.ylabel('PC 1')
plt.grid()
plt.show()
plt.savefig('Grafici/clust_on_time_HKK.png')