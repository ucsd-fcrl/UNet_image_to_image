import numpy as np
import os
import functions_collection as ff
import Defaults

cg = Defaults.Parameters()

class Build():
    def __init__(self):
        self.a = 1

    def __build__(self):
        # find all the simulated data
        simulated_data_list = ff.sort_timeframe(ff.find_all_target_files(['*/random_*'],cg.moving_data_dir),0,'_')

        x_list = []
        y_list = []

        for i in range(0,len(simulated_data_list)):
            # print(simulated_data_list[i])
            x_file = ff.find_all_target_files(['simulated/recon.nii.gz'],simulated_data_list[i])
            if len(x_file) != 1:
                ValueError('no randomly simulated data')
            x_list.append(x_file[0])

            patient_id = os.path.basename(os.path.dirname(simulated_data_list[i]))
            y_file = ff.find_all_target_files(['partial/*.nii.gz'],os.path.join(cg.static_data_dir,patient_id))
            if len(y_file) != 1:
                ValueError('no corresponding original data')
            y_list.append(y_file[0])

        return np.asarray(x_list), np.asarray(y_list)

        