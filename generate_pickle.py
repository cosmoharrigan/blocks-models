"""
Creates pickle files containing the single-block dataset:
input.pkl containing 8080 rows of the form block_x,block_y,block_z,block_type
output.pkl containing 8080 rows, where each is a vector of length 32x32
"""

__author__ = 'Cosmo Harrigan'

import numpy as np
import scipy.ndimage

input_csv = '/Users/cosmo/mldata/single-block-dataset-1/june30/processed/single-block-dataset-1.csv'
output_folder='/Users/cosmo/mldata/single-block-dataset-1/june30/grayscale_128x128/'

input_file = '/Users/cosmo/blocks-models/input.pkl'
output_file = '/Users/cosmo/blocks-models/minecraft_single_block_images_grayscale_128x128.pkl'

IMAGE_SIZE = 128*128

INCLUDE_BLOCK_TYPE = True
BLOCK_TYPE_ONLY = False

# convert the csv file
if BLOCK_TYPE_ONLY:
    input_data = np.genfromtxt(input_csv, delimiter=',', skip_header=0, names=True,
                               usecols='block_type',
                               dtype=np.int64)
elif INCLUDE_BLOCK_TYPE:
    input_data = np.genfromtxt(input_csv, delimiter=',', skip_header=0, names=True,
                               usecols=('block_x',
                                        'block_y',
                                        'block_z',
                                        'block_type'),
                               dtype=np.int64)
else:
    input_data = np.genfromtxt(input_csv, delimiter=',', skip_header=0, names=True,
                           usecols=('block_x',
                                    'block_y',
                                    'block_z'),
                           dtype=np.int64)

# convert structured array to ndarray
input_data_processed = np.zeros([8080, 4], dtype=np.float64)
input_data_processed = input_data.view(np.int64).reshape(input_data.shape + (-1,)).astype('float64')

input_data_processed.dump(input_file)

# convert the images
images = np.zeros([8080, IMAGE_SIZE], dtype=np.float64)
for i in range(8080):
    image_idx = i+1  # 1-based indexing
    image_filename = output_folder + str(image_idx) + '.png'
    images[i] = scipy.ndimage.imread(image_filename, flatten=True, mode="RGB").flatten().astype('float64')

images.dump(output_file)
