import torch
import torch.nn as nn

import os
import numpy as np

from utils import FK, load, save, Quaternions
from torch.utils.data import Dataset, DataLoader

## Training hyper parameters
epoch = 50
batch_size = 128
learning_rate = 1e-3

## Model parameters
model_rotation_type = 'q'

### Your implementation here
class MotionInterpolationModel(nn.Module):
    def __init__(self, input_feature_size):
        super(MotionInterpolationModel, self).__init__()
        
    
    def forward(self, x):
        
        return 


class MotionDataset(Dataset):
    def __init__(self, motion_folder, rotation_type,  model_clip_size, model_clip_offset, is_train):
        rotation_set, root_positon_set, self.motion_files = [], [], []
        self.is_train = is_train
        for file in os.listdir(motion_folder):
            if not file.split('.')[1] == 'bvh':
                continue
            bvh_file_path = '%s/%s' % (motion_folder, file)
            return_eular = False if rotation_type == 'q' else True
            rotations, positions, offsets, parents, names, frametime = load(filename=bvh_file_path, return_eular=return_eular)
            rotation_set.append(rotations.qs if rotation_type == 'q' else rotations)
            root_positon_set.append(positions[:, 0])
            self.motion_files.append(file)
            if is_train:
                mirrored_rotations, mirrored_root_position = self.mirroring(rotation_set[-1], root_positon_set[-1])
                rotation_set.append(mirrored_rotations)
                root_positon_set.append(mirrored_root_position)
                self.motion_files.append('mirrored_' + file)
        self.offsets, self.parents, self.names, self.frametime = offsets, parents, names, frametime
        self.rotations, self.file_idx = self.chunking(rotation_set, chunk_size=model_clip_size, offset=model_clip_offset, target_fps=30)
        self.root_positon, _ = self.chunking(root_positon_set, chunk_size=model_clip_size, offset=model_clip_offset, target_fps=30)
        self.rotations_noised = self.noising(self.rotations, rotation_type)
        self.joint_number = rotation_set[0].shape[1]
        self.rotation_number = 4 if rotation_type == 'q' else 3

    def chunking(self, data, chunk_size, offset, target_fps):
        res = []
        file_idx = []
        for item_idx, item in enumerate(data):
            sampling_factor = int(1/self.frametime/target_fps)
            item = item[0:item.size:sampling_factor]
            filename = self.motion_files[item_idx]
            for start_idx in np.arange(0, item.shape[0] - chunk_size - 1, offset):
                file_idx.append(filename)
                res.append(item[start_idx:start_idx+chunk_size].astype(np.float32))
        return res, file_idx

    def noising(self, data, rotation_type):
        res = []
        for item_idx, item in enumerate(data):
            if rotation_type == 'q':
                noises = np.random.normal(0, 0.02, size=item.shape)
            else:
                noises = np.random.normal(0, 0.02*np.radians(90), size=item.shape)
            res.append((item + noises).astype(np.float32))
        return res

    def mirroring(self, rotation, root_position):
        mirrored_rotations = rotation.copy()
        morrored_root_position = root_position.copy()
        joints_left = [1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22, 23]
        joints_right = [6, 7, 8, 9, 10, 24, 25, 26, 27, 28, 29, 30]
        mirrored_rotations[:, joints_left] = rotation[:, joints_right]
        mirrored_rotations[:, joints_right] = rotation[:, joints_left]
        mirrored_rotations[:, :, [2, 3]] *= -1
        morrored_root_position[:, 0] *= -1
        return mirrored_rotations, morrored_root_position

    def __len__(self):
        assert(len(self.rotations) == len(self.root_positon))
        return len(self.rotations)

    def __getitem__(self, idx):    
        return self.rotations[idx], self.rotations_noised[idx], self.root_positon[idx], self.file_idx[idx]

    def __get_feature_number__(self):
        return self.rotations[0].shape[1]*self.rotations[0].shape[2]


if __name__ == '__main__':
    model_clip_size = 15
    model_clip_offset = 2
    train_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, model_clip_size, model_clip_offset, is_train=True)
    test_dataset = MotionDataset('./data/edin_locomotion_valid', model_rotation_type, model_clip_size, model_clip_size, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    joint_number, rotation_number = train_dataset.joint_number, train_dataset.rotation_number
    model = MotionInterpolationModel(input_feature_size=test_dataset.__get_feature_number__())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    OFFSET = 5
    keyframes = np.arange(0, model_clip_size, OFFSET)
    if torch.cuda.is_available():
        model.cuda()

    for epoch_idx in range(epoch):
        print_freq = len(train_dataloader) // 10
        model.train()
        for batch_idx, (batch_rotations, _, batch_root_positions, file_name) in enumerate(train_dataloader):
            ### Your implementation here
            print

