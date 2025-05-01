import scipy.io as sio
import numpy as np

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
index_dict = {}

for layer in layers:
    features_path = f'/flash/DoyaU/shuhei/bdata_decoding/features/{layer}/n01729977_653.mat'
    mat_contents = sio.loadmat(features_path)

    # Get flattened feature vector
    image_feature = mat_contents['feat'][0].flatten()

    # Randomly select 1000 indices
    print(image_feature.shape[0])
    selected_index = np.random.choice(image_feature.shape[0], 1000, replace=False)

    # Store in dict
    index_dict[layer] = selected_index.astype(np.int32)  # int32 for MATLAB compatibility

# Save as 'index' field into a .mat file
sio.savemat('feature_index_random1000.mat', {'index': index_dict})
