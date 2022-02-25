import torch
import torch.nn as nn

import os
import subprocess
import numpy as np

from utils import FK, load, save, Quaternions
from torch.utils.data import Dataset, DataLoader

## Training hyper parameters
epoch = 100
batch_size = 128
learning_rate = 1e-3

## Model parameters
model_rotation_type = 'q'

class MotionInterpolationModel(nn.Module):
    def __init__(self, input_feature_size):
        super(MotionInterpolationModel, self).__init__()
        
    
    def forward(self, x):
        
        return 


class MotionDataset(Dataset):
    def __init__(self, motion_folder, rotation_type,  model_clip_size, model_clip_offset, is_train):
        rotation_set, root_positon_set, self.motion_files = [], [], []
        for file in os.listdir(motion_folder):
            a = ('test' in file)
            if file.split('.')[1] == 'bvh':
                if 'test' in file and not is_train:
                    model_clip_offset = model_clip_size
                    pass
                elif 'test' not in file and is_train:
                    pass
                else:
                    continue
            bvh_file_path = '%s/%s' % (motion_folder, file)
            return_eular = False if rotation_type == 'q' else True
            rotations, positions, offsets, parents, names, frametime = load(filename=bvh_file_path, return_eular=return_eular)
            rotation_set.append(rotations.qs if rotation_type == 'q' else rotations)
            root_positon_set.append(positions[:, 0])
            self.motion_files.append(file)
        self.offsets, self.parents, self.names, self.frametime = offsets, parents, names, frametime
        self.rotations, self.file_idx = self.chunking(rotation_set, chunk_size=model_clip_size, offset=model_clip_offset, target_fps=30)
        self.root_positon, _ = self.chunking(root_positon_set, chunk_size=model_clip_size, offset=model_clip_offset, target_fps=30)
        self.rotations_noised = self.noising(self.rotations)
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

    def noising(self, data):
        res = []
        for item_idx, item in enumerate(data):
            noises = np.random.normal(-np.radians(30), np.radians(30), size=item.shape)
            res.append(item + noises)
        return res

    def __len__(self):
        assert(len(self.rotations) == len(self.root_positon))
        return len(self.rotations)

    def __getitem__(self, idx):
        return self.rotations[idx], self.rotations_noised[idx], self.root_positon[idx], self.file_idx[idx]

    def __get_feature_number__(self):
        return self.rotations[0].shape[1]*self.rotations[0].shape[2]


if __name__ == '__main__':
    model_clip_size = 60
    model_clip_offset = 5
    train_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, model_clip_size, model_clip_offset, is_train=True)
    test_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, model_clip_size, model_clip_offset, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    joint_number, rotation_number = train_dataset.joint_number, train_dataset.rotation_number
    model = MotionInterpolationModel(input_feature_size=test_dataset.__get_feature_number__())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if torch.cuda.is_available():
        model.cuda()

    for epoch_idx in range(epoch):
        print_freq = len(train_dataloader) // 10

        model.train()
        for batch_idx, (batch_rotations, _, batch_root_positions, file_name) in enumerate(train_dataloader):
            batch_input_starting = batch_rotations[:20].reshape((20, batch_rotations.shape[1], -1)).transpose(1, 2)
            batch_input_ending = batch_rotations[40:].reshape((20, batch_rotations.shape[1], -1)).transpose(1, 2)
            batch_target = batch_rotations[20:40].reshape((20, batch_rotations.shape[1], -1)).transpose(1, 2)

            if torch.cuda.is_available():
                batch_input_starting, batch_input_ending, batch_target = batch_input_starting.cuda(), batch_input_ending.cuda(), batch_target.cuda()

           