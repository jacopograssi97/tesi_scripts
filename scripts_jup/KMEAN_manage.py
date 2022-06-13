import numpy as np
from eofs.xarray import Eof
from sklearn.cluster import KMeans
from scipy import signal 
import matplotlib.pyplot as plt
import xarray as xr



def kmean_models_dataset(dataset, dataset_names, n_season, n_iter):
        
        """

        """

        model_list = []
        clust_centers = []

        for i in np.arange(0,len(dataset_names),1):

            data = getattr(dataset, dataset_names[i]).to_numpy()
            model, clust_center = kmean_training(data, n_season, n_iter)

            b = xr.DataArray(clust_center)

            model_list.append(model)
            clust_centers.append(b.rename(f'cc_{dataset_names[i]}'))

        cc_database = xr.merge(clust_centers)

        return model_list, cc_database


def kmean_prediction_dataset(dataset, dataset_names, model_list):
        
    """

    """

    predictions = []

    for i in np.arange(0,len(dataset_names),1):

        data = getattr(dataset, dataset_names[i]).to_numpy()
        prediction = kmean_preditting(model_list[i], data)

        a = xr.DataArray(prediction, dims=['time'])

        predictions.append(a.rename(f'prediction_{dataset_names[i]}'))


    predict_database = xr.merge(predictions)
    predict_database = xr.merge([predict_database,dataset])

    return predict_database



def kmean_preditting(model, to_fit):
        
        """
        """

        prediction = model.predict(to_fit)

        return prediction



def kmean_training(training_set, n_clust, max_iter):
        
        """
        """

        model = KMeans(n_clusters=n_clust, max_iter=max_iter).fit(training_set)

        clust_centers = model.cluster_centers_
    
        return model, clust_centers