# UNet_image_to_image
**Author: Zhennong Chen, PhD**<br />

This repo builds a simple U-Net, which takes a CT image (e.g., motion-corrupted image) as the input and outputs another CT  image (e.g., motion-free image). The whole pipeline (training and prediction) is written for 2D CT images (only takes 2D images as the input), but the U-Net (CNN.py) can be used for both 2D and 3D inputs, and you can easily change the pipeline script for 3D.

**This repo is compatible with tensorflow-gpu 2.4.1 and cuda 11.0.3.**

Files:<br />
- ```CNN.py```: Architecture of UNet<br />
- ```Generator.py```: Datagenerator, use class inherited from tensorflow.keras.utils.Sequence<br />
- ```set_defaults.sh``` and ```Defaults.py```: Preset parameters for the experiment<br />
- ```Data_processing.py```: image processing to adapt for the model<br />
- ```Build_list.py```: generate input and output file list<br />
- ```main_train.py```: train the model<br />
- ```main_predict.py```: predict on new cases via the trained model<br />

To run this repo:<br />
- step 0: change the file path as you need<br />
- step 1: run ```. ./set_defaults.sh```<br />
- step 2: run ```python main_train.py```<br />
- step 3: run ```python main_predict.py```<br />
