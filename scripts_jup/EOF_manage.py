import numpy as np
from eofs.xarray import Eof
import xarray as xr
import json 


def eof_base_interface(dataset_to_base, dataset_names_to_base,json_filename):
            
        
        with open(json_filename, 'r') as openfile:
  
                # Reading from json file
                json_EOF = json.load(openfile)

        eof_base_dataset, pc_base_dataset, exp_dataset, solver_list = eof_base_dataset_create(dataset_to_base, dataset_names_to_base, 
                                                                                                json_EOF['weights'], json_EOF['center'], 
                                                                                                json_EOF['ddof'], json_EOF['eofscaling'], 
                                                                                                json_EOF['pcscaling'], json_EOF['n_comp'])

                                                                
        return eof_base_dataset, pc_base_dataset, exp_dataset, solver_list


def eof_proj_interface(solver_list, dataset_to_project, dataset_names, json_filename):
            
        
        with open(json_filename, 'r') as openfile:
  
                # Reading from json file
                json_EOF = json.load(openfile)

        pc_proj_dataset = eof_projection_database(solver_list, dataset_to_project,
                                                                 dataset_names, json_EOF['n_comp'])

                                                                
        return pc_proj_dataset






def eof_base_dataset_create(dataset, dataset_names, weights, center, ddof, eofscaling, pcscaling, n_comp):
        
        """

        """

        eofs = []
        pcs = []
        exp_vars = []
        solvers = []

        for i in np.arange(0,len(dataset_names),1):

                eof, pc, exp_var, solver = eof_base(to_base = getattr(dataset, dataset_names[i]), weights = weights, 
                                                center = center, ddof = ddof, eofscaling = eofscaling, pcscaling = pcscaling,
                                                 n_comp = n_comp)

                eofs.append(eof.rename(f'{dataset_names[i]}'))
                pcs.append(pc.rename(f'{dataset_names[i]}'))
                exp_vars.append(exp_var.rename(f'{dataset_names[i]}'))
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




def eof_projection_database(solver_list, dataset_to_project, dataset_names, n_comp):
        
        """

        """

        pc_projection = []

        for i in range(len(dataset_names)):

                supp = getattr(dataset_to_project, dataset_names[i]) - getattr(dataset_to_project, dataset_names[0]).mean('time')
                pc_projection.append(solver_list[i].projectField(supp, neofs = n_comp).rename(f'{dataset_names[i]}'))

        pc_proj_dataset = xr.merge(pc_projection)


     
        return pc_proj_dataset

