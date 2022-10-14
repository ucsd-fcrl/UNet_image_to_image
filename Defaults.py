# System
import os

class Parameters():

  def __init__(self):
  
    # # Number of partitions in the crossvalidation.
    # self.num_partitions = int(os.environ['CG_NUM_PARTITIONS'])
    
    # Dimension of padded input, for training.
    self.dim = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']))
    self.slice_num = int(os.environ['CG_CROP_Z'])

    self.unetdim = len(self.dim)
  
    # Seed for randomization.
    self.seed = int(os.environ['CG_SEED'])
      
  
    # UNet Depth
    self.unet_depth = int(os.environ['CG_UNET_DEPTH']) # default = 5
    # Feature map 
    self.conv_depth = [16 * (2** x) for x in range(self.unet_depth)] + [int(16 * (2** (self.unet_depth - 1)) * (0.5** (x+1))) for x in range(self.unet_depth - 1)]
    print(self.conv_depth)
  
    # How many images should be processed in each batch?
    self.batch_size = int(os.environ['CG_BATCH_SIZE'])

    # How many cases should be read in each batch?
    self.patients_in_one_batch = int(os.environ['CG_PATIENTS_IN_ONE_BATCH'])
  
    # # Translation Range
    # self.xy_range = float(os.environ['CG_XY_RANGE'])
  
    # # Scale Range
    # self.zm_range = float(os.environ['CG_ZM_RANGE'])

    # # Rotation Range
    # self.rt_range=float(os.environ['CG_RT_RANGE'])
  
    # Should Flip
    self.flip = False

    # Total number of epochs to train
    self.epochs = int(os.environ['CG_EPOCHS'])
    self.lr_epochs = int(os.environ['CG_LR_EPOCHS'])
    self.initial_power = int(os.environ['CG_INITIAL_POWER'])


    # # folders
    # for VR dataset
    self.main_data_dir = os.environ['CG_MAIN_DATA_DIR']
    self.static_data_dir = os.environ['CG_MOTION_FREE_DATA_DIR']
    self.moving_data_dir = os.environ['CG_MOVING_DATA_DIR']
    self.model_dir = os.environ['CG_MODEL_DIR']
    self.predict_dir = os.environ['CG_PREDICT_DATA_DIR']