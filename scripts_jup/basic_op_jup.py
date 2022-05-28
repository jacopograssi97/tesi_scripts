import numpy as np
from eofs.xarray import Eof
from sklearn.cluster import KMeans 




def eof_base(to_base, weights, center, ddof, eofscaling, pcscaling, n_comp):
        
        """

        """
        
        # Create an EOF solver to do the EOF analysis
        solver = Eof(array = to_base, weights = weights, center = center, ddof = ddof)

        # Retrieve the EOF and the pc
        # Options for scaling:
        # 0 -> Un-scaled EOFs (default)
        # 1 -> EOFs are divided by the square-root of their eigenvalue
        # 2 -> EOFs are multiplied by the square-root of their eigenvalue
        eof = solver.eofs(eofscaling = eofscaling)
        pc  = solver.pcs(pcscaling = pcscaling)

        # Retrieving expressed variance for each component
        exp_var = solver.varianceFraction()

        # Selecting only the wanterd number of components
        eof = eof.sel(mode=slice(0,n_comp-1))
        pc = pc.sel(mode=slice(0,n_comp-1))
        exp_var = exp_var.sel(mode=slice(0,n_comp-1))

        return eof, pc, exp_var, solver




def eof_project(solver, to_project, n_comp):
        
        """

        """
        pc_projection = solver.projectField(to_project, neofs = n_comp)

        return pc_projection



def kmean_training(training_set, n_clust, max_iter):
        
        """
        """

        model = KMeans(n_clusters=n_clust, max_iter=max_iter).fit(training_set)

        clust_centers = model.cluster_centers_
    
        return model, clust_centers




def kmean_preditting(model, to_fit):
        
        """
        """

        prediction = model.predict(to_fit)

        return prediction





def gregorian_day(xarr):
    
    """
    """

    m = xarr['time.month']
    d = []
    k = 1

    for i in np.arange(0,len(m),1):

        if i == len(m)-1:
            k = k+1
            break

        elif m[i+1] < m[i]:
            d.append(k)
            k = 1

        else:
            d.append(k)
            k = k+1

    d.append(k)

    return np.array(d)