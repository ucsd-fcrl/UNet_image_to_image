#!/usr/bin/env python

import Defaults
import CNN
import Generator
import Data_processing as dp
import functions_collection as ff
import Build_list

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

cg = Defaults.Parameters()

trial_name = 'simple_trial'

# image processing
# patient_list = ff.find_all_target_files(['*'],cg.static_data_dir)
img_list = ff.find_all_target_files(['*/original/*.nii.gz'],cg.static_data_dir)
dp.save_partial_volumes(img_list,slice_range = None)

# build lists
print('Build List...')
b = Build_list.Build()
x_list, y_list = b.__build__()
print(x_list,y_list)
print(x_list.shape)
# define train and validation
x_list_trn = x_list[0:8]; y_list_trn = y_list[0:8]
x_list_val = x_list[8:10]; y_list_val = y_list[8:10]
print(x_list_trn.shape, x_list_val.shape)

# create model
print('Create Model...')
input_shape = cg.dim + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]
levels, ds, us, final_feature,final_image = CNN.get_CNN(cg.dim ,cg.conv_depth,layer_name='cnn')(model_inputs[0])
model_outputs += [final_image]
model = Model(inputs = model_inputs,outputs = model_outputs)

load_model = False; model_file = []
if load_model == True:
    model.load_weights(model_file)

# compile model
print('Compile Model...')
opt = Adam(lr = 1e-4)
losses={'cnn':'MAE'} 
model.compile(optimizer= opt, 
              loss= losses,
              metrics= {'cnn':'MAE',})

# set callbacks
print('Set callbacks...')
model_fld = os.path.join(cg.model_dir,trial_name,'models')
model_name = 'model-' + trial_name + '-CNN'
filepath=os.path.join(model_fld,  model_name +'-{epoch:03d}.hdf5')
ff.make_folder([os.path.dirname(model_fld), model_fld, os.path.join(os.path.dirname(model_fld), 'logs')])
csv_logger = CSVLogger(os.path.join(os.path.dirname(model_fld), 'logs',model_name + '_training-log.csv')) # log will automatically record the train_accuracy/loss and validation_accuracy/loss in each epoch
callbacks = [csv_logger,
                ModelCheckpoint(filepath,          
                                monitor='val_loss',
                                save_best_only=False,),
                 LearningRateScheduler(CNN.learning_rate_step_decay),   # learning decay
                ]


# Fit
print('Fit model...')
datagen = Generator.DataGenerator(x_list_trn,y_list_trn,patient_num = x_list_trn.shape[0], 
                                slice_num = cg.slice_num, 
                                batch_size = cg.batch_size, 
                                patients_in_one_batch = cg.patients_in_one_batch, 
                                shuffle = True, 
                                normalize = True,
                                adapt_shape = [cg.dim[0],cg.dim[1],cg.slice_num],
                                input_channels = 1 , 
                                output_channels = 1,
                                seed = 5
                                 )

valgen = Generator.DataGenerator(x_list_val,y_list_val,patient_num = x_list_val.shape[0], 
                                slice_num = cg.slice_num, 
                                batch_size = cg.batch_size, 
                                patients_in_one_batch = cg.patients_in_one_batch, 
                                shuffle = True, 
                                normalize = True,
                                adapt_shape = [cg.dim[0],cg.dim[1],cg.slice_num],
                                input_channels = 1 , 
                                output_channels = 1,
                                seed = 5
                                 )

model.fit_generator(generator = datagen,
                    epochs = cg.epochs,
                    validation_data = valgen,
                    callbacks = callbacks,
                    verbose = 1,
                    )

