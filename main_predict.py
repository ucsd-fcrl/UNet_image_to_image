#!/usr/bin/env python

import Defaults
import CNN
import Generator
import Data_processing as dp
import functions_collection as ff
import Build_list

import os
import numpy as np
import nibabel as nb
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

cg = Defaults.Parameters()

trial_name = 'simple_trial'
epoch_pick = '200'
model_file = ff.find_all_target_files(['*'+epoch_pick+'.hdf5'],os.path.join(cg.model_dir,trial_name,'models'))
print(model_file[0])
model_file = model_file[0]
save_folder = os.path.join(cg.predict_dir,trial_name)
ff.make_folder([save_folder])

# build lists
print('Build List...')
b = Build_list.Build()
x_list, y_list = b.__build__()
# define prediction
# x_list_predict = x_list[8:10]; y_list_predict = y_list[8:10]
x_list_predict = x_list[8:10]; y_list_predict = y_list[8:10]
print(x_list_predict.shape, x_list_predict)

# create model and load weights
print('Create Model and Load Weights...')
input_shape = cg.dim + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]
levels, ds, us, final_feature,final_image = CNN.get_CNN(cg.dim ,cg.conv_depth,layer_name='cnn')(model_inputs[0])
model_outputs += [final_image]
model = Model(inputs = model_inputs,outputs = model_outputs)

load_model = True
if load_model == True:
    model.load_weights(model_file)


# predict generator
print('Predict...')

for i in range(0,x_list_predict.shape[0]):
    patient_id_1 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(x_list_predict[i]))))
    patient_id_2 = os.path.basename(os.path.dirname(os.path.dirname(x_list_predict[i])))
    print(patient_id_1, patient_id_2)

    datagen = Generator.DataGenerator(np.asarray([x_list_predict[i]]),np.asarray([y_list_predict[i]]),
                                patient_num = 1,
                                slice_num = cg.slice_num, 
                                batch_size = cg.slice_num, 
                                patients_in_one_batch = 1, 
                                shuffle = False, 
                                normalize = True,
                                adapt_shape = [cg.dim[0],cg.dim[1],cg.slice_num],
                                input_channels = 1 , 
                                output_channels = 1,
                                seed = 5,
                                 )
    pred = model.predict_generator(datagen, verbose = 1, steps = 1,)
    print(pred.shape,np.mean(pred),np.max(pred),np.min(pred))
    
    x = nb.load(x_list_predict[i])
    affine = x.affine
    x = x.get_fdata()
    
    pred = pred[...,0]
    pred = pred * 1000# return to the original value before normalization
    pred = np.rollaxis(pred,0,3)
    pred = dp.crop_or_pad(pred,x.shape)
    print(pred.shape,np.mean(pred),np.max(pred),np.min(pred))

    nb_pred = nb.Nifti1Image(pred,affine)
    filename = os.path.join(save_folder,patient_id_1,patient_id_2,'pred.nii.gz')
    ff.make_folder([os.path.dirname(os.path.dirname(filename)),os.path.dirname(filename)])
    nb.save(nb_pred, filename)
    ff.save_grayscale_image(pred[...,int(pred.shape[-1]/2)].T, os.path.join(save_folder,patient_id_1,patient_id_2,'pred.png'))

    
    

    
