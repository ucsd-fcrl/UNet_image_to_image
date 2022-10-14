# tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import random
import Data_processing
import Defaults
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math

cg = Defaults.Parameters()


class DataGenerator(Sequence):

    def __init__(self,X,Y,
        patient_num = None, 
        slice_num = None, 
        batch_size = None, 
        patients_in_one_batch = None , 
        shuffle = None,
        normalize = None,
        adapt_shape = None,
        input_channels = 1,
        output_channels = 1,
        seed = 10):

        self.X = X
        self.y = Y
        # self.image_data_generator = image_data_generator
        self.patient_num = patient_num
        self.slice_num = slice_num
        self.batch_size = batch_size
        self.patients_in_one_batch = patients_in_one_batch
        self.shuffle = shuffle
        self.normalize = normalize
        self.adapt_shape = adapt_shape
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.seed = seed

        self.slice_in_one_batch = int(self.batch_size / self.patients_in_one_batch)

        self.on_epoch_end()
        
    def __len__(self):
        return self.X.shape[0] * self.slice_num // self.batch_size

    def on_epoch_end(self):
        
        self.seed += 1
        # print('seed is: ',self.seed)

        patient_list = np.random.permutation(self.patient_num)
        index_array = []
        for p in patient_list:
            
            if self.shuffle == True:
                slices= np.random.permutation(self.slice_num)
            else:
                slices = np.arange(self.slice_num)

            for s in slices:
                index_array.append([p,s])

        if self.shuffle == True: # put several cases into one batch instead of just one case
            new_index_array = []
            slices_in_one_group = self.patients_in_one_batch * self.slice_num
            for i in range(0,int(self.patient_num / self.patients_in_one_batch)):
                g = index_array[slices_in_one_group * i:slices_in_one_group * (i+1)]
                random.shuffle(g)
                new_index_array.extend(g)
            
            index_array = new_index_array
                
        self.indices = np.asarray(index_array)
        # print('all indexes: ', self.indices,len(self.indices))

    def __getitem__(self,index):
        'Generate one batch of data'

        'Generate indexes of the batch'
        total_slice = self.patient_num * self.slice_num
        current_index = (index * self.batch_size) % total_slice
        if total_slice > current_index + self.batch_size:   # the total number of cases is adequate for next loop
            current_batch_size = self.batch_size
        else:
            current_batch_size = total_slice - current_index  # approaching to the tend, not adequate, should reduce the batch size


        indexes = self.indices[current_index : current_index + current_batch_size]
        if self.shuffle == True:
            indexes = indexes[indexes[:,0].argsort()]
        # print('indexes in this batch: ',indexes)

        # allocate memory
        batch_x = np.zeros(tuple([current_batch_size]) + tuple([self.adapt_shape[0],self.adapt_shape[1]]) + tuple([self.input_channels]))
        batch_y= np.zeros(tuple([current_batch_size]) + tuple([self.adapt_shape[0],self.adapt_shape[1]]) + tuple([self.output_channels]))
        
        volume_already_load = []
        load = False
        for i,j in enumerate(indexes):
            case = j[0]
            # Is it a new case so I need to load the image volume?
            if i == 0:
                volume_already_load.append(case)
                load = True
                # print('let us start', i, case, volume_already_load[0],load)
            else:
                if case == volume_already_load[0]:
                    load = False
                    # print(i, case, volume_already_load[0],load)
                else:
                    load = True
                    volume_already_load[0] = case
                    # print(i, case, volume_already_load[0],load)
                
            if load == True:
                x = self.X[case]
                # print('now loading x : ',x)
                y = self.y[case]
                # print('now loading y : ',y)
                x = Data_processing.adapt(x,self.adapt_shape)
                y = Data_processing.adapt(y,self.adapt_shape)
                
                # print(x.shape,y.shape)
                
            # print('which slice?: ',j[1])
            img_x = x[:,:,j[1],:]
            img_y = y[:,:,j[1],:]
            

            if self.normalize == True:
                img_x = Data_processing.normalize_image(img_x)
                img_y = Data_processing.normalize_image(img_y)

           

            batch_x[i] = img_x
            batch_y[i] = img_y

        # if self.normalize == True:
        #     batch_x = Data_processing.normalize_image(batch_x)
        #     batch_y = Data_processing.normalize_image(batch_y)

        return batch_x, batch_y