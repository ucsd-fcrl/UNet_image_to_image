import numpy as np
import nibabel as nb
import os
import Defaults
import functions_collection as ff

cg = Defaults.Parameters()

def crop_or_pad(array, target, value=0):
    """
    Symmetrically pad or crop along each dimension to the specified target dimension.
    :param array: Array to be cropped / padded.
    :type array: array-like
    :param target: Target dimension.
    :type target: `int` or array-like of length array.ndim
    :returns: Cropped/padded array. 
    :rtype: array-like
    """
    # Pad each axis to at least the target.
    margin = target - np.array(array.shape)
    padding = [(0, max(x, 0)) for x in margin]
    array = np.pad(array, padding, mode="constant", constant_values=value)
    for i, x in enumerate(margin):
        array = np.roll(array, shift=+(x // 2), axis=i)

    if type(target) == int:
        target = [target] * array.ndim

    ind = tuple([slice(0, t) for t in target])
    return array[ind]


def adapt(x, target):
  x = nb.load(x).get_data()
  # clip the very high value
  x = crop_or_pad(x, target)
  x = np.expand_dims(x, axis = -1)
#   print('after adapt, shape of x is: ', x.shape)
  return x


def normalize_image(x):
    # a common normalization method in CT
    # if you use (x-mu)/std, you need to preset the mu and std
    
    return x.astype(np.float32) / 1000


def save_partial_volumes(img_list,slice_range = None): # only save some slices of an original CT volume
    for img_file in img_list:
        print(img_file)
        f = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'partial')
        if os.path.isfile(os.path.join(f,os.path.basename(img_file))) == 1:
            print('already saved')
            continue

        x = nb.load(img_file)
        img = x.get_data()

        if slice_range == None:
            slice_range = [int(img.shape[-1]/2) - 30, int(img.shape[-1]/2) + 30]
        
        img = img[:,:,slice_range[0]:slice_range[1]]

        ff.make_folder([f])
        img = nb.Nifti1Image(img,x.affine)
        nb.save(img, os.path.join(f,os.path.basename(img_file)))
