import numpy as np
import netCDF4 as nc
from cdo import Cdo
from scipy.fftpack import fft
cdo=Cdo()
import pandas as pd
from eofs.standard import Eof
from sklearn.cluster import KMeans
from scipy import signal

def extract_nc(file_in, field, extract_time, units_time, calendar):
    
    """
    This function extracts the data from a .nc file
    """

    database = nc.Dataset(file_in)
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
        time = nc.num2date(time, units_time, calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        time = pd.to_datetime(time)

    return output, latitude, longitude, time


def altitude_mask(field, orography, bottom_level, top_level):
    
    """
    """
    
    masked_field = np.ma.masked_where(orography<bottom_level,field)
    masked_field = np.ma.masked_where(orography>top_level,masked_field)
    
    return masked_field


def eof_custom(field_3d, n_eof, n_pc, report):
        
        """
        """

        # Rescaling using anomalies
        field_3d = field_3d - field_3d.mean(axis=0)

        # Create an EOF solver to do the EOF analysis
        solver = Eof(field_3d)

        # Retrieve the EOF and the pc
        eof = solver.eofs(eofscaling=0)
        pc  = solver.pcs(pcscaling=0)
        exp_var = solver.varianceFraction()

        eof = np.squeeze(eof[0:n_eof,:,:])
        pc = np.squeeze(pc[:,0:n_pc])
        
        if report == True:

            print()
            print('─' * 40)
            print('EOF REPORT:')
            print()
            print('Selected output dimension: eof -> ' + str(n_eof) + '   pc -> ' + str(n_pc))
            print('Input field size: ' + str(np.shape(field_3d)))
            print('Output eof size: ' + str(np.shape(eof)))
            print('Output pc size:  ' + str(np.shape(pc)))
            print('Explained variance for each dimension: ' + str(exp_var[0:n_eof]*100))
            print('─' * 40)

        return eof, pc, exp_var



def kmean_custom(to_clust, n_clust, max_iter,  report):
        
        """
        """

        model = KMeans(n_clusters=n_clust, max_iter=max_iter).fit(to_clust)
        clust = model.predict(to_clust)
        clust_centers = model.cluster_centers_
    
        if report == True:

            print()
            print('─' * 40)
            print('K-MEANS REPORT:')
            print()
            print('Selected number of cluster: ' + str(n_clust))
            print('Input data size: ' + str(np.shape(to_clust)))
            print(' Centers - N° elements: ')
            
            for i in range(n_clust):
                print('Cluster ' + str(i+1) + ': ' + str(clust_centers[i,:]) + ' - ' + str(np.count_nonzero(clust == i)))

            print('─' * 40)

        return clust, clust_centers



def periodogram_custom(time_serie, samp_int, nfft,  report):
        
    """
    """

    time_serie_norm = (time_serie-np.mean(time_serie))/np.std(time_serie)
    f, Pxx = signal.periodogram(np.squeeze(time_serie_norm), samp_int, nfft=nfft)

    f = f[2:int(nfft/2+1)]
    Pxx = Pxx[2:int(nfft/2+1)]

    period = 1./f
    
    if report == True:

        print()
        print('─' * 40)
        print('PERIODOGRAM REPORT:')
        print()
        print('Selected number of frequences: ' + str(nfft))
        print('Selected sampling interval: ' + str(samp_int))
        print('─' * 40)

    return Pxx, period, f