import subprocess
import numpy as np

from utils import *

'''
Your implement here: Implement more interpolation algorithms on motion concatenation task
To-Do:
Complete the apply_interpolation function with two interpolation algorithms
* Students are encouraged to use existing libraries for this, like SciPy
* Not necessarily Bezier, such as nearest, nearest-up, slinear, quadratic which can be chosen
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
    elif interpolation_method == 'slerp':

        print
    else:
        raise NotImplementedError('No support interpolation way %s' % interpolation_method)
    return data_filled


bvh_file_path1 = './data/motion_walking.bvh'
bvh_file_path2 = './data/motion_basket.bvh' 
end_frame = 1200
start_frame = 0

betweening_number = 100

rotations1, positions1, offsets, parents, names, frametime = load(filename=bvh_file_path1)
rotations2, positions2, offsets, parents, names, frametime = load(filename=bvh_file_path2)

rotation_clip1, position_clip1 = rotations1[:end_frame].qs, positions1[:end_frame]
rotation_clip2, position_clip2 = rotations2[start_frame:].qs, positions2[start_frame:]

betweening_rotation = np.zeros((betweening_number, rotation_clip1.shape[1], 4))
betweening_rotation[0] = rotation_clip1[-1]
betweening_rotation[-1] = rotation_clip2[0]

for joint_index in range(rotations1.shape[1]):
    betweening_rotation[:, joint_index, :] = apply_interpolation(betweening_rotation[:, joint_index], 
                                                            keyframes=[0, betweening_number-1], 
                                                            interpolation_method='linear')

betweening_position = np.zeros((betweening_number, rotation_clip1.shape[1], 3))
betweening_position[0] = position_clip1[-1]
betweening_position[-1] = position_clip2[0]
betweening_position[:, 0, :] = apply_interpolation(betweening_position[:, 0], 
                                                            keyframes=[0, betweening_number-1],
                                                            interpolation_method='linear')

output_file_path = './data/walking_concat_basket.bvh'

new_rotation = np.concatenate([rotation_clip1, betweening_rotation, rotation_clip2])
new_position = np.concatenate([position_clip1, betweening_position, position_clip2])

save(output_file_path, Quaternions(new_rotation).normalized(), new_position, offsets, parents, names, frametime)
# Comment on this line if it's hard to configure the blender into you system enviroment
subprocess.call('blender -P load_bvhs.py -- -r %s -c %s -o' % (output_file_path, output_file_path), shell=True)
# subprocess.call('blender -P load_bvhs.py -- -r %s -c %s --render' % (bvh_file_path, output_file_path), shell=True)