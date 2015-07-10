import numpy as np

input_file = '/Users/cosmo/blocks-models/input.pkl'
output_file = '/Users/cosmo/blocks-models/minecraft_single_block_images_grayscale_128x128.pkl'

input_data = np.load(input_file)
print('input:\n', input_data)
print('input shape:', input_data.shape)

output_data = np.load(output_file)
print('output:\n', output_data)
print('output shape: ', output_data.shape)
