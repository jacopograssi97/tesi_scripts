from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cpf
import netCDF4 as nc
from cdo import Cdo
from scipy import signal
import math



cdo=Cdo()
plt.rcParams["font.family"] = "Times New Roman"

# Selecting input file
infile="/work/datasets/obs/ERA5/total_precipitation/mon/ERA5_total_precipitation_mon_0.25x0.25_sfc_1950-2020.nc"

# Selecting Hindu-Kush Karakoram region (west)
region_HKK = cdo.sellonlatbox(71,78,32,37, input=infile)

# Selecting Himalayan region (east)
region_Him = cdo.sellonlatbox(78,93,25,32, input=infile)

# Mean field for both regions
mean_fld_HKK = cdo.timmean(input=region_HKK)
mean_fld_Him = cdo.timmean(input=region_Him)

# Time serie for both regions
time_serie_HKK = cdo.fldmean(input=region_HKK)
time_serie_Him = cdo.fldmean(input=region_Him)

# Climatology
climat_HKK = cdo.ymonmean(input=time_serie_HKK)
climat_Him = cdo.ymonmean(input=time_serie_Him)

## MEAN MAP
# Opening tmp file (.nc file)
database = nc.Dataset(mean_fld_HKK)
mean_fld_HKK = database['tp'][:]*1000
latitude_HKK = database['lat'][:]
longitude_HKK = database['lon'][:]
database.close()

database = nc.Dataset(mean_fld_Him)
mean_fld_Him = database['tp'][:]*1000
latitude_Him = database['lat'][:]
longitude_Him = database['lon'][:]
database.close()

fig = plt.figure(figsize=(8, 6), dpi=300, edgecolor='k')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([70, 94, 24, 38], ccrs.Geodetic())
ax.add_patch(mpatches.Rectangle(xy=[71, 32], width=7, height=5, edgecolor='k', linewidth=3, facecolor=None, fill=False,  transform=ccrs.PlateCarree()))
ax.add_patch(mpatches.Rectangle(xy=[78, 25], width=15, height=7, edgecolor='k', linewidth=3, facecolor=None, fill=False,  transform=ccrs.PlateCarree()))
ax.imshow(np.squeeze(mean_fld_HKK), origin='upper', extent=[71,78,32,37], transform=ccrs.PlateCarree(),cmap='jet', vmax=0.27, vmin=0)
ax.imshow(np.squeeze(mean_fld_Him), origin='upper', extent=[78,93,25,32], transform=ccrs.PlateCarree(),cmap='jet', vmax=0.27, vmin=0)
ax.text(71.5, 31, 'A', color='k', size=26, ha='center', va='center', transform=ccrs.PlateCarree())
ax.text(91.5, 33, 'B', color='k', size=26, ha='center', va='center', transform=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ax.add_feature(cpf.BORDERS)
plt.title("Mean precipitation 1950-2020")
plt.show()
plt.savefig('map.png')

# Climate
database = nc.Dataset(climat_HKK)
climat_HKK = database['tp'][:]*1000
climat_HKK = np.squeeze(climat_HKK)
database.close()

database = nc.Dataset(climat_Him)
climat_Him = database['tp'][:]*1000
climat_Him = np.squeeze(climat_Him)
database.close()

fig, axs = plt.subplots(2,figsize=(8, 6), dpi=300, edgecolor='k')
plt.suptitle("Average of each calendar month")

axs[0].plot(climat_HKK, label='HKK', color='r')
axs[1].plot(climat_Him, label='Himalaya', color='b')

for ax in axs.flat:
    ax.set(xlabel='Month', ylabel='Tp', ylim=[0, 0.5])
    ax.grid()
    ax.legend()

for ax in axs.flat:
    ax.label_outer()

plt.show()
plt.savefig('climatology.png')

# Periodogram
database = nc.Dataset(time_serie_HKK)
time_serie_HKK = database['tp'][:]*1000
time_serie_HKK = np.squeeze(time_serie_HKK)
database.close()

database = nc.Dataset(time_serie_Him)
time_serie_Him = database['tp'][:]*1000
time_serie_Him = np.squeeze(time_serie_Him)
database.close()


nfft = 2048
nfft_2 = int(nfft/2 +1)
samp_int = 12

time_serie_norm_HKK = (time_serie_HKK-np.mean(time_serie_HKK))/np.std(time_serie_HKK)
f_HKK, Pxx_HKK = signal.periodogram(np.squeeze(time_serie_norm_HKK), samp_int, nfft=nfft)
period_HKK = 1./f_HKK

time_serie_norm_Him = (time_serie_Him-np.mean(time_serie_Him))/np.std(time_serie_Him)
f_Him, Pxx_Him = signal.periodogram(np.squeeze(time_serie_norm_Him), samp_int, nfft=nfft)
period_Him = 1./f_Him


fig, axs = plt.subplots(2,figsize=(8, 6), dpi=300, edgecolor='k')
plt.suptitle("Periodogram")

axs[0].plot((period_HKK[2:nfft_2]), Pxx_HKK[2:nfft_2], label='HKK', color='r')
axs[1].plot((period_Him[2:nfft_2]), Pxx_Him[2:nfft_2], label='Himalaya', color='b')

for ax in axs.flat:
    ax.set(xlabel='Period [yr]', ylabel='PSD', ylim=[0, 50], xlim=[0, 2])
    ax.grid()
    ax.legend()

for ax in axs.flat:
    ax.label_outer()

plt.show()
plt.savefig('periodogram.png')

cdo.cleanTempDir()