from re import M
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import cartopy.feature as cpf
from cdo import Cdo
cdo=Cdo()
from basic_op import *
import bottleneck as bn
import pandas as pd

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

pr = cdo.ydaymean(input = region_prec)

# Extracting data and operating transormation
prec_fields, latitude, longitude, time = extract_nc(pr, 'tp', True, 'hours since 1950-01-01', 'proleptic_gregorian')
prec_fields = np.squeeze(prec_fields*1000*24)

eof, pc, exp_var = eof_custom(prec_fields, 4, 4, True)
to_fit = np.squeeze(pc)

#to_fit = (to_fit - np.mean(to_fit, axis=0))/np.std(to_fit,axis=0)

clust, clust_centers = kmean_custom(to_fit, 4, 300, True)

g_d = gregorian_day(time)

m = time.month
a = np.arange(0,len(m),1)

for i in np.arange(0,len(m),1):
    if m[i]==1:
        a[i]=1
    elif m[i]==2:
        a[i]=1
    elif m[i]==3:
        a[i]=2
    elif m[i]==4:
        a[i]=2
    elif m[i]==5:
        a[i]=2
    elif m[i]==6:
        a[i]=3
    elif m[i]==7:
        a[i]=3
    elif m[i]==8:
        a[i]=3
    elif m[i]==9:
        a[i]=4
    elif m[i]==10:
        a[i]=4
    elif m[i]==11:
        a[i]=4
    elif m[i]==12:
        a[i]=1


fig = plt.figure(figsize=(8, 6), dpi=300, edgecolor='k')
plt.scatter(to_fit[:,0], to_fit[:,1], c=clust, cmap='jet' )
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Daily data - colors by season')
plt.grid()
plt.show()
plt.savefig('Grafici/KClust/clust_on_time_HKK_daily_m.png')