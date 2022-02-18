import subprocess
import numpy as np

from utils import *

'''
Your implement here: Implement more interpolation algorithms
To-Do:
Calculate the joint position by the hierarchical structure and joint rotation 
* The rotation on parent joint will impact all child joint
* Any rotation library is allowed to use, like scipy
* You can add parameters with your own design
'''
def apply_interpolation(data, keyframes, interpolation_method):
    data_filled = data.copy()
    if interpolation_method == 'linear':
        start_idx = keyframes[0]
        for end_idx in keyframes[1:]:
            start_data, end_data = data[start_idx], data[end_idx]
            in_fills = end_idx - start_idx
            base_incrementation = (end_data - start_data)/(in_fills)
            data_filled[start_idx:end_idx] = np.arange(0, in_fills)[:, np.newaxis]*base_incrementation[np.newaxis, :].repeat(in_fills, axis=0) + data_filled[start_idx]
            # A slower way but easier to understand:
            # for times in range(0, end_idx - start_idx):
            #     data_filled[start_idx + times] = start_data + base_incrementation*times
            start_idx = end_idx
    elif interpolation_method == 'bezier':
        
        print
    elif interpolation_method == 'bspline':

        print 
    else:
        raise NotImplementedError('No support interpolation way %s' % interpolation_method)
    return data_filled


OFFSET = 120
bvh_file_path = './data/motion_walking.bvh'
rotations, positions, offsets, parents, names, frametime = load(filename=bvh_file_path)
rotations_fake, positions_fake = np.zeros_like(rotations.qs), np.zeros_like(positions)

keyframes = np.arange(1, rotations.shape[0], OFFSET)

rotations_fake[keyframes] = rotations.qs[keyframes]
positions_fake[keyframes] = positions[keyframes]

for joint_index in range(rotations.shape[1]):
    rotations_fake[:, joint_index, :] = apply_interpolation(rotations_fake[:, joint_index], 
                                                            keyframes=keyframes, 
                                                            interpolation_method='linear')
positions_fake[:, 0, :] = apply_interpolation(positions_fake[:, 0], 
                                                            keyframes=keyframes,
                                                            interpolation_method='linear')

output_file_path = '%s_interpolate_%s.bvh' % (bvh_file_path[:-4], OFFSET)
save(output_file_path, Quaternions(rotations_fake).normalized(), positions_fake, offsets, parents, names, frametime)
subprocess.call('blender -P load_bvhs.py -- -r %s -c %s' % (bvh_file_path, output_file_path), shell=True)
# subprocess.call('blender -P load_bvhs.py -- -r %s -c %s --render' % (bvh_file_path, output_file_path), shell=True)