import numpy as np
from eofs.xarray import Eof
import xarray as xr
import json 


def eof_base_interface(dataset,json_filename):
            
        
        with open(json_filename, 'r') as openfile:
  
                # Reading from json file
                json_EOF = json.load(openfile)

        eof_database, pc_database, exp_database, solver_database = eof_base_database(dataset, json_EOF['dataset_to_base'], json_EOF['weights'], json_EOF['center'], json_EOF['ddof'], json_EOF['eofscaling'], json_EOF['pcscaling'], json_EOF['n_comp'])

                                                                
        return eof_database, pc_database, exp_database, solver_database





def eof_base_database(dataset, dataset_to_base, weights, center, ddof, eofscaling, pcscaling, n_comp):
        
        """

        """

        eofs = []
        pcs = []
        exp_vars = []
        solvers = []

        for i in np.arange(0,len(dataset_to_base),1):

                eof, pc, exp_var, solver = eof_base(to_base = getattr(dataset, dataset_to_base[i]), weights = weights, center = center, ddof = ddof, eofscaling = eofscaling, pcscaling = pcscaling, n_comp = n_comp)

                eofs.append(eof.rename(f'eof_{dataset_to_base[i]}'))
                pcs.append(pc.rename(f'pc_{dataset_to_base[i]}'))
                exp_vars.append(exp_var.rename(f'exp_var_{dataset_to_base[i]}'))
                solvers.append(solver)

        eof_database = xr.merge(eofs)
        pc_database = xr.merge(pcs)
        exp_database = xr.merge(exp_vars)

     

        return eof_database, pc_database, exp_database, solvers





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




def eof_projection_database(solver_database, database_to_base, dataset_to_project, n_comp):
        
        """

        """

        pcs_projection = []

        for i in np.arange(0,len(dataset_to_project),1):

                pc_projection = getattr(solver_database, database_to_base[0]).projectField(dataset_to_project, neofs = n_comp)

                pcs_projection.append(pc_projection.rename(f'pc_proj_{dataset_to_project[i]}'))

        pc_proj_database = xr.merge(pcs_projection)

     
        return pc_proj_database


















def eof_project(solver, to_project, n_comp):
        
        """

        """
        pc_projection = solver.projectField(to_project, neofs = n_comp)

        return pc_projection