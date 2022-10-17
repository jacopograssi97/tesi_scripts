from random import randint
import numpy as np


class Radially_Constrained_Cluster(object):

    def __init__(self, data_to_cluster, n_seas, n_iter = 1000, learning_rate = 1, l_r_scheduler = False, scheduling_factor = 1, len_consistancy_check = False, min_len = 1, mode = 'single', n_ensemble = 1000, s_factor = 0.1):

        '''
            -> data to cluster: time series with timesteps on first dimension and features on second
            -> n_iter: number of iterations
            -> n_seas: number of clusters
            -> learning_rate: maximun number of day for stochastic breakpoints upgrade 
            -> l_r_scheduler: if True the learning rate is reduced at each iteration that improves the metrics
            -> scheduling_factor: factor for reducing learning_rate
            -> len_consistancy_check: if True each season length is bounded have a minimun length
            -> min_len: minimum length for bounded seasonal length

        '''

        self.l_r_scheduler = l_r_scheduler
        self.len_consistancy_check = len_consistancy_check

        # Establishing the len of the serie
        self.len_serie = np.size(data_to_cluster,axis=0)
        self.data_to_cluster = data_to_cluster


        # Check parameter consistance
        if len_consistancy_check == True:

            if self.len_serie/n_seas < min_len:

                raise ValueError(f'Cannot create {n_seas} season of {min_len} days. Please check your input parameters')
            
            else:

                self.n_seas = n_seas
                self.min_len = min_len

        else:

            self.n_seas = n_seas
            self.min_len = min_len

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.scheduling_factor = scheduling_factor
        self.mode = mode
        self.s_factor = s_factor
        self.n_ensemble = n_ensemble




    def fit(self):

        if self.mode == 'single':

            self.breakpoints, self.centroid_story, self.error_story =  self.single_fit()


        if self.mode == 'ensemble_stochastic':

            err = []
            bd = []

            for j in range(self.n_ensemble):

                self.min_len = self.min_len + randint(-int(self.min_len/self.s_factor),int(self.min_len/self.s_factor))
                self.learning_rate = self.learning_rate + randint(-int(self.learning_rate/self.s_factor),int(self.learning_rate/self.s_factor))
                self.scheduling_factor = self.scheduling_factor + randint(-int(self.scheduling_factor/self.s_factor),int(self.scheduling_factor/self.s_factor))
                self.n_iter = self.n_iter + randint(-int(self.n_iter/self.s_factor),int(self.n_iter/self.s_factor))

                self.breakpoints, self.centroid_story , self.error_story =  self.single_fit()

                err.append(self.error_story[self.n_iter-1])
                bd.append(np.sort(self.breakpoints))
            
            self.breakpoints = np.int32(np.mean(bd,axis=0))







    def single_fit(self):

        # Defining list for metrics saving
        cc_tot = []
        error_tot = []

        # Main loop
        for j in range(self.n_iter):

            # Generating random starting breackpoints - equally distributed over time (firt iteration)
            if j == 0:
                upgrade, b = generate_starting_bpoints(self.n_seas, self.min_len, self.len_serie)

            # Randomly upgrading breakpoints in the range breakpoint +- learning rate (other iteration)
            else:
                upgrade, b = upgrade_breakpoints(self.n_seas, b, self.learning_rate, self.len_serie)

            # Generating index for each season
            idx = generate_season_idx(self.n_seas, b, self.len_serie)

            # Control on min season length - if false is skipped
            if self.len_consistancy_check == True:
                len_ok = check_season_len(self.n_seas, idx, self.min_len)

            # If control is sett on false is skipped
            else:
                len_ok = True

            # Case all season lengths are ok -> computing metrics
            if len_ok == True:
                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)
                cc_tot.append(centroids)
                error_tot.append(np.sum(error))

                # Skipping first iteration
                if j > 0:
                    # Checking if the breakpoints upgrade has improved the metrics
                    if error_tot[j]>error_tot[j-1]:
                        # If not downgrade breakpoints on last iteration
                        b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)

                    # Scheduling learning rate for best minimun localization
                    elif (error_tot[j-1] - error_tot[j-2]) < 0 and self.l_r_scheduler==True and self.learning_rate > 1:
                        self.learning_rate = schedule_learning_rate(self.learning_rate, self.scheduling_factor)


            # If there are too short seasons just pretend like nothing happend
            # Downgrading breakpoints to previous iteration
            else:
                b = downgrade_breakpoints(self.n_seas, b, upgrade, self.len_serie)
                idx = generate_season_idx(self.n_seas, b, self.len_serie)
                centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)
                cc_tot.append(centroids)
                error_tot.append(np.sum(error))

        return np.sort(np.int32(b)), np.float64(cc_tot), np.float64(error_tot)




    def get_prediction(self):

        # Converting breakpoints in a time series 
        prediction = np.zeros((self.len_serie,1))

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        for i in range(self.n_seas):
            prediction[idx[i]] = i

        return prediction


    def get_final_error(self):

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

        return np.sum(error)
    
    
     
    def get_centroids(self):

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        centroids, error = compute_metrics(self.n_seas, self.data_to_cluster, idx)

        return centroids
        

    def get_index(self):

        idx = generate_season_idx(self.n_seas, self.breakpoints, self.len_serie)

        return idx





def generate_starting_bpoints(n_season,min_len,len_serie):

    to_select = np.arange(min_len,len_serie,1)

    b_start = []
    upgrade = []

    for i in range(n_season):

        if i == 0:

            b_start.append(int((len_serie-1)/n_season))
            upgrade.append(0)

        else:
        
            b_start.append(b_start[i-1]+int((len_serie-1)/n_season))
            upgrade.append(0)

        if b_start[i] > len_serie-1:

            b_start[i] = b_start[i]-len_serie-1

    b_start = np.sort(b_start)

    return upgrade, b_start



def generate_season_idx(n_season, b, len_serie):

    idx = []


    if n_season == 1:

        idx.append(np.arange(0, len_serie, 1))

    

    else:



        for i in np.arange(-1, n_season-1,1):

            if b[i]>b[i+1]:

                idx_0 = np.arange(b[i], len_serie, 1)
                idx_1 = np.arange(0, b[i+1], 1)
                idx.append(np.concatenate((idx_0, idx_1), axis=None))
            

            else:
 
                idx.append(np.arange(b[i], b[i+1],1))


    return idx


def compute_metrics(n_season, data_to_cluster, idx):

    centroids = []
    error = []

    for i in range(n_season):
                
        centroids.append(np.nanmean(data_to_cluster[idx[i]], axis = 0))
        error.append(np.nansum(np.power(data_to_cluster[idx[i]]-centroids[i],2), axis = 0))

            

    return centroids, error


def upgrade_breakpoints(n_season, old_b, learning_rate, len_serie):

    upgrade = []
    new_b = []

    for k in range(n_season):

        upgrade.append(randint(-learning_rate,learning_rate))
            
        new_b.append(old_b[k]+upgrade[k])

        if new_b[k]>len_serie-1:

            new_b[k]=new_b[k]-len_serie-1

        if new_b[k]<0:

            new_b[k]=len_serie-1+new_b[k]

    return upgrade, np.array(new_b)



def downgrade_breakpoints(n_season, new_b, upgrade, len_serie):

    old_b = []

    for k in range(n_season):

        old_b.append(new_b[k]-upgrade[k])

        if old_b[k]>len_serie-1:

            old_b[k]=old_b[k]-len_serie-1

        if old_b[k]<0:

            old_b[k]=len_serie-1+old_b[k]

    return np.array(old_b)



def schedule_learning_rate(learning_rate, scheduling_factor):

    return np.int32(learning_rate/scheduling_factor)




def check_season_len(n_season, idx, min_len):

    len_ok = True

    for k in range(n_season):

        if len(idx[k])<min_len:

            len_ok = False

    return len_ok