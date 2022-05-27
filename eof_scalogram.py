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
from scipy import signal
from obspy.signal.tf_misfit import cwt


eof_d = [0,1,2,3,4]
reg = ['HKK']

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
                pc = (pc-np.mean(pc))/np.std(pc)

                t, dt = np.linspace(0, 71, len(pc), retstep=True)
                fs = 1/dt
                w = 8.
                freq = np.linspace(1, 5, 100)
                widths = w*fs / (2*freq*np.pi)
                cwtm = signal.cwt(pc, signal.morlet2, widths, w=w)


                fig = plt.figure(figsize=(4, 5), dpi=300, edgecolor='k')
                gs = fig.add_gridspec(3, 1)
                plt.suptitle('EOF' + str(eof_dim+1) + '  region: ' + region + '  exp var: ' + str("{:.2f}".format(exp_var[eof_dim]*100)) + '%')

                ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

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

                ax2 = fig.add_subplot(gs[1, :])
                ax2.plot(t, pc)
                ax2.set_ylabel('PC' + str(eof_dim+1))
                ax2.grid()

                ax3 = fig.add_subplot(gs[2, :])
                im  = ax3.pcolormesh(t, freq, np.abs(cwtm), cmap='jet', shading='gouraud')
                ax3.set_ylabel('F [yr^-1]')
                ax3.set_xlabel('Time [yr since 1950]')
                ax3.grid()

                plt.savefig('Grafici/Eof/Eof_scal/scalogram_eof' + str(eof_dim+1) + '_' + region + '.png')