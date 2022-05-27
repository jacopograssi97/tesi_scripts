from os import system
from matplotlib.cbook import report_memory
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import cartopy.feature as cpf
from cdo import Cdo
cdo=Cdo()
from basic_op import *


eof_d = [0,1,2,3,4]
reg = ['HKK', 'Him', 'all']

report = True

system('clear')

# Selecting netCDF files 
file_in_prec = '~/work/jacopo/DATA/ERA5_prec_day_HKKH.nc'

for eof_dim in eof_d:

        for region in reg:

                print()
                print('Processing:  region -> ' + region + ',  eof component -> ' + str(eof_dim+1))
                print()

                if region == 'HKK':        

                        region_prec = cdo.sellonlatbox(71, 78, 32, 37, input = file_in_prec)

                if region == 'Him':        

                        region_prec = cdo.sellonlatbox(78, 93, 25, 32, input = file_in_prec)

                if region == 'all':        

                        region_prec = cdo.sellonlatbox(71,94,21,39, input = file_in_prec)

                
                # Extracting data and operating transormation
                prec_fields, latitude, longitude, time = extract_nc(region_prec, 'tp', True, 'hours since 1950-01-01', 'proleptic_gregorian')
                prec_fields = np.squeeze(prec_fields*1000*24)

                eof, pc, exp_var = eof_custom(prec_fields, eof_dim+2, eof_dim+2, report)

                pc = np.squeeze(pc[:,eof_dim])

                pxx, period, f = periodogram_custom(pc, 365, 2048, report)


                fig = plt.figure(figsize=(6, 8), dpi=300, edgecolor='k')
                gs = fig.add_gridspec(4, 3)
                plt.suptitle('EOF' + str(eof_dim+1) + '  region: ' + region + '  exp var: ' + str("{:.2f}".format(exp_var[eof_dim]*100)) + '%')


                ax1 = fig.add_subplot(gs[0:1, :], projection=ccrs.PlateCarree())
                ax1.set_extent([70, 94, 24, 38], ccrs.Geodetic())

                if region == 'HKK':        

                        im = ax1.imshow(np.squeeze(eof[eof_dim,:,:]), origin='upper', extent=[71, 78, 32, 37], transform=ccrs.PlateCarree(),cmap='jet')

                if region == 'Him':        

                        im = ax1.imshow(np.squeeze(eof[eof_dim,:,:]), origin='upper', extent=[78,93,25,32], transform=ccrs.PlateCarree(),cmap='jet')

                if region == 'all':        

                        im = ax1.imshow(np.squeeze(eof[eof_dim,:,:]), origin='upper', extent=[71,94,21,39], transform=ccrs.PlateCarree(),cmap='jet')

                ax1.coastlines(resolution='auto', color='k')
                ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=False)
                ax1.add_feature(cpf.BORDERS)
                plt.colorbar(im)

                ax2 = fig.add_subplot(gs[2, :])
                ax2.plot(time, pc)
                ax2.set_ylabel('PC' + str(eof_dim+1))
                ax2.set_xlabel('Time')
                ax2.grid()

                ax3 = fig.add_subplot(gs[3, :])
                ax3.plot(period, pxx)
                ax3.set_ylabel('PSD')
                ax3.set_xlabel('Period [yr]')
                ax3.grid()

                plt.savefig('Grafici/Eof/period_eof' + str(eof_dim+1) + '_' + region + '.png')