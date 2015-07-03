"""
Preprocessing step to convert the high-resolution color images to
low-resolution grayscale images before dimensionality reduction is applied
"""

import scipy.ndimage
import scipy.misc
import numpy as np

import glob
filenames = {}
# filenames = glob.glob('/Users/cosmo/mldata/single-block-dataset-1/june30/processed/*.png')
filenames = glob.glob('/Users/cosmo/mldata/single-block-dataset-1/june30/processed/grayscale/*.png')

# grayscale_downsampled_folder = '/Users/cosmo/mldata/single-block-dataset-1/june30/processed/grayscale/'
grayscale_downsampled_folder = '/Users/cosmo/mldata/single-block-dataset-1/june30/grayscale_tiny/'

i = 0
for filename in filenames:
    original_image = scipy.ndimage.imread(filename, flatten=True)
    downsampled_image = scipy.misc.imresize(original_image, size=(32, 32))
    
    extracted_filename = filename.split('/')[-1]
    print(grayscale_downsampled_folder + extracted_filename)
    scipy.misc.imsave(grayscale_downsampled_folder + extracted_filename, downsampled_image)

    print('Processed image #{0}'.format(i))
    i += 1
