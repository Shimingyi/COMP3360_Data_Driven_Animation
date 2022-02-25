from statistics import mode
import torch
import torch.nn as nn

import os
import subprocess
import numpy as np

from utils import FK, load
from torch.utils.data import Dataset, DataLoader

## Training hyper parameters
epoch = 100
batch_size = 128
learning_rate = 1e-3

## Model parameters
model_clip_size = 1
model_clip_offset = 1
model_rotation_type = 'euler'
model_fc_layers = 3
# model_fc_channels = 1024


class MotionDenoisingModel(nn.Module):
    def __init__(self, input_feature_size):
        super(MotionDenoisingModel, self).__init__()
        

    def forward(self, x):
        
        return 


class MotionDataset(Dataset):
    def __init__(self, motion_folder, rotation_type, is_train):
        rotation_set, root_positon_set, self.motion_files = [], [], []
        for file in os.listdir(motion_folder):
            a = ('test' in file)
            if file.split('.')[1] == 'bvh':
                if 'test' in file and not is_train:
                    pass
                elif 'test' not in file and is_train:
                    pass
                else:
                    continue
            bvh_file_path = '%s/%s' % (motion_folder, file)
            rotations, positions, offsets, parents, names, frametime = load(filename=bvh_file_path)
            rotation_set.append(rotations.qs if rotation_type == 'q' else rotations.euler())
            root_positon_set.append(positions[:, 0])
            self.motion_files.append(file)
        self.offset, self.parent, self.names, self.frametime = offsets, parents, names, frametime
        self.rotations, self.file_idx = self.chunking(rotation_set, chunk_size=model_clip_size, offset=model_clip_offset, target_fps=30)
        self.root_positon, _ = self.chunking(root_positon_set, chunk_size=model_clip_size, offset=model_clip_offset, target_fps=30)
        self.rotations_noised = self.noising(self.rotations)

    def chunking(self, data, chunk_size, offset, target_fps):
        res = []
        file_idx = []
        for item_idx, item in enumerate(data):
            sampling_factor = int(1/self.frametime/target_fps)
            item = item[0:item.size:sampling_factor]
            for start_idx in np.arange(0, item.shape[0] - chunk_size - 1, offset):
                file_idx.append(item_idx)
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
        return self.rotations[idx], self.rotations_noised[idx], self.root_positon[idx]

    def __get_feature_number__(self):
        return self.rotations[0].shape[1]*self.rotations[0].shape[2]


if __name__ == '__main__':
    model_clip_size = 1
    model_clip_offset = 1
    train_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, is_train=True)
    test_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = MotionDenoisingModel(input_feature_size=train_dataset.__get_feature_number__())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    if torch.cuda.is_available():
        mode.cuda()

    for epoch_idx in range(epoch):
        for batch_idx, (batch_rotations, batch_rotations_noised, batch_root_positions) in enumerate(train_dataloader):
            batch_input = batch_rotations_noised.reshape(batch_rotations.shape[0], -1)
            batch_target = batch_rotations.reshape(batch_rotations.shape[0], -1)

            ### Your implementation here

