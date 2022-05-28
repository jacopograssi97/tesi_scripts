from cdo import Cdo
import json
import xarray as xr
cdo=Cdo()


def input_interface(json_filename):

    with open(json_filename, 'r') as openfile:
  
        # Reading from json file
        json_IO = json.load(openfile)

    step1 = time_interpolation(json_IO['files_in'], json_IO['start_date'], json_IO['start_time'], json_IO['increment_time'])
    step2 = time_selection(step1, json_IO['start_date'], json_IO['start_time'], json_IO['end_date'])
    step3 = region_selection(step2, json_IO['western_longitude_limit'], json_IO['easter_longitude_limit'], json_IO['down_latitude_limit'], json_IO['up_latitude_limit'])
    step4 = region_regridding(step3, json_IO['interp_method'])
    step5 = merge_dataset(step4, json_IO['variable_name'], json_IO['dataset_names'])

    cdo.cleanTempDir()

    return step5



def merge_dataset(files_in, variable_name, dataset_names):

    data_set = []
    data_array = []

    for i in range(0, len(files_in)):

        data_set.append(xr.open_dataset(files_in[i]))

        data_array.append(getattr(data_set[i], variable_name[i]).rename(dataset_names[i]))

    final_dataset = dt = xr.merge(data_array)

    return final_dataset




def time_interpolation(files_in, start_date, start_time, increment_time):

    time_interpolated = []

    for i in range(0, len(files_in)):
        time_interpolated.append(cdo.inttime(start_date, start_time, increment_time,
                        input = files_in[i]))
                            
    return time_interpolated




def time_selection(files_in, start_date, start_time, end_date):

    time_selected = []

    start_date_comp = f"{start_date}T{start_time}"
    end_date_comp = f"{end_date}T{start_time}"

    for i in range(0, len(files_in)):
        time_selected.append(cdo.seldate(start_date_comp, end_date_comp,
                        input = files_in[i]))
                            
    return time_selected


def region_selection(files_in, western_longitude_limit, easter_longitude_limit, down_latitude_limit, up_latitude_limit):

    region_selected = []

    for i in range(0, len(files_in)):
        region_selected.append(cdo.sellonlatbox(western_longitude_limit, easter_longitude_limit, 
                            down_latitude_limit, up_latitude_limit, 
                            input = files_in[i]))
                                
    return region_selected



def region_regridding(files_in, interp_method):

    files_in_reg = []

    for i in range(0, len(files_in)):
        if interp_method == 'bilinear':
            files_in_reg.append(cdo.remapbil(files_in[0], input = files_in[i]))

        elif interp_method == 'bicubic':
            files_in_reg.append(cdo.remapbic(files_in[0], input = files_in[i]))

        elif interp_method == 'distance_weighted':
            files_in_reg.append(cdo.remapdis(files_in[0], input = files_in[i]))

        elif interp_method == 'nearest_neigbhor':
            files_in_reg.append(cdo.remapnn(files_in[0], input = files_in[i]))

        elif interp_method == 'first_order_cons':
            files_in_reg.append(cdo.remapcon(files_in[0], input = files_in[i]))

        elif interp_method == 'secondo_order_cons':
            files_in_reg.append(cdo.remapcon2(files_in[0], input = files_in[i]))

        elif interp_method == 'largest_area_frac':
            files_in_reg.append(cdo.remaplaf(files_in[0], input = files_in[i]))

    return files_in_reg

