import torch
import torch.nn as nn
import numpy as np

from utils import load


bvh_file_path = './data/motion_walking.bvh'
rotations, positions, offsets, parents, names, frametime = load(filename=bvh_file_path)

print('The shape of rotation variable is %s' % str(rotations.qs.shape))         # (2752, 31, 4)
print('The shape of positions variable is %s' % str(positions.shape))           # (2752, 31, 3)

## nn.Linear example
input = torch.zeros((128, 3, 100)) # batch_size, channel, feature
fc_layer = nn.Linear(in_features=100, out_features=10)
output = fc_layer(input)    # (128, 3, 10)

rotations_tensor = torch.from_numpy(np.array(rotations.qs, dtype=np.float32))
fc_layer = nn.Linear(in_features=4, out_features=10)

# (2752, 31, 10)

output = fc_layer(rotations_tensor)

## nn.Conv1D example
input = torch.zeros((128, 3, 100))
conv1d_layer = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=2, stride=2)
output = conv1d_layer(input)

# (128, 16, 50)

input = torch.zeros((128, 3, 100))
# 3 - feature information 
# 100 - sequence information 

# (2752, 31, 4)
# batch 
# 31*4 - feature
# 2752 - sequence information 

# (2752, 31, 4)
rotations_tensor = torch.from_numpy(np.array(rotations.qs, dtype=np.float32)) 

# (1, 60, 31*4)
rotations_tensor = rotations_tensor[:60].reshape((1, 60, 124))

# (1, 31*4, 60)
rotations_tensor = rotations_tensor.transpose(1, 2)
conv1d_layer = nn.Conv1d(in_channels=124, out_channels=256, kernel_size=2, stride=2)
output = conv1d_layer(rotations_tensor)

## Encoder
rotations_tensor = torch.from_numpy(np.array(rotations.qs, dtype=np.float32))  # (2752, 31, 4)
rotations_tensor_item = rotations_tensor[0].reshape((1, 124))
fc_layer1 = nn.Linear(in_features=124, out_features=96)
fc_layer2 = nn.Linear(in_features=96, out_features=72)
fc_layer3 = nn.Linear(in_features=72, out_features=54)

output1 = fc_layer1(rotations_tensor_item)
output2 = fc_layer2(output1)
output3 = fc_layer3(output2)

## Decoder
# fc_layer4 = nn.Linear(in_features=54, out_features=124)

fc_layer4 = nn.Linear(in_features=54, out_features=72)
fc_layer5 = nn.Linear(in_features=72, out_features=96)
fc_layer6 = nn.Linear(in_features=96, out_features=124)

output4 = fc_layer4(output3)
output5 = fc_layer5(output4)
output6 = fc_layer6(output5)

loss = nn.MSELoss()(output6, rotations_tensor_item)

print