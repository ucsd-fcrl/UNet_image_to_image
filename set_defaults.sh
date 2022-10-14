## to run this in terminal, type:
# chmod +x set_defaults.sh
# . ./set_defaults.sh   

## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="0"

# volume dimension
export CG_CROP_X=624 # has to be divisible by 2^(unet_depth-1),e.g., when unet_depth = 5, divisible by 2^4 = 16
export CG_CROP_Y=624 # same as above
export CG_CROP_Z=60 # 2D experiment, don't need to be divisible


# set the batch:
# in one batch, the model will read n slices from N patients
export CG_PATIENTS_IN_ONE_BATCH=2  # set it larger will slow down the training speed.
# n should be divisible by CG_CROP_Z, let's set n = 15

# batch_size = N * n
export CG_BATCH_SIZE=30 # N = 2 patients in one batch, n = 15 slices from each patient


# set U-NET feature depth
export CG_UNET_DEPTH=5 # filter number = [16,32,64,128,256,128,64,32,16]

# set learning epochs
export CG_EPOCHS=200
export CG_LR_EPOCHS=40 # the number of epochs for learning rate change 
export CG_INITIAL_POWER=-3

# set random seed
export CG_SEED=10

########## need to define it if doing segmentation
# # set number of partitioning groups, = 5 means we will do 5-fold cross-validation
# export CG_NUM_PARTITIONS=5


########## need to define it if need augmentation
# # set data augmentation range
# export CG_XY_RANGE="0.1"   #0.1
# export CG_ZM_RANGE="0.1"  #0.1
# export CG_RT_RANGE="10"   #15


# folders for Zhennong's dataset (change based on your folder paths)
export CG_MAIN_DATA_DIR="/workspace/Documents/data/CT_motion/CT_images/"
export CG_MOTION_FREE_DATA_DIR="${CG_MAIN_DATA_DIR}example_CT_volume/"       
export CG_MOVING_DATA_DIR="${CG_MAIN_DATA_DIR}simulated_data/"      
export CG_MODEL_DIR="/workspace/Documents/data/CT_motion/Models/"
export CG_PREDICT_DATA_DIR="${CG_MAIN_DATA_DIR}predicted_data/"