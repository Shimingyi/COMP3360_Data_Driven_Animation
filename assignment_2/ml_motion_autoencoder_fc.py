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

class MotionAutoEncoder_FC(nn.Module):
    def __init__(self, input_feature_size):
        super(MotionAutoEncoder_FC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_feature_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 96),
            nn.ReLU(True), 
            nn.Linear(96, 72),
            nn.ReLU(True), 
            nn.Linear(72, 48)
        )
        self.decoder = nn.Sequential(
            nn.Linear(48, 72),
            nn.ReLU(True),
            nn.Linear(72, 96),
            nn.ReLU(True), 
            nn.Linear(96, 128),
            nn.ReLU(True), 
            nn.Linear(128, input_feature_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstrcution = self.decoder(latent)
        return latent, reconstrcution


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
    model_clip_size = 1
    model_clip_offset = 1
    train_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, model_clip_size, model_clip_offset, is_train=True)
    test_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, model_clip_size, model_clip_offset, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    joint_number, rotation_number = train_dataset.joint_number, train_dataset.rotation_number
    
    model = MotionAutoEncoder_FC(input_feature_size=train_dataset.__get_feature_number__())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if torch.cuda.is_available():
        model.cuda()


    # 1000 images
    # batch-level-loop: 10 images as one 1 batch, 100 batches   <-   loop on 100 batches
    # epoch-level-loop: 100 / 50 / 20

    for epoch_idx in range(epoch):
        print_freq = len(train_dataloader) // 10

        # (128, 124*5)  ->  (128, 124*5)
        # (128, 124, 5)   

        # 128, 124, 5     ->       128, 124, 10    ->   128, 124, 16 

        model.train()
        for batch_idx, (batch_rotations, _, batch_root_positions, file_name) in enumerate(train_dataloader):   
            batch_input = batch_rotations.reshape(batch_rotations.shape[0], -1)
            batch_target = batch_rotations.reshape(batch_rotations.shape[0], -1)

            if torch.cuda.is_available():
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            deep_latent, output = model(batch_input)
            
            loss = torch.nn.MSELoss()(output, batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % print_freq == 0:
                print('Training: Epoch %d, (%s/%s) the reconstuction loss is %04f' % (epoch_idx, batch_idx, len(train_dataloader), loss.item()))

        model.eval()
        errors = []
        for batch_idx, (batch_rotations, _, batch_root_positions, file_name) in enumerate(test_dataloader):   
            batch_input = batch_rotations.reshape(batch_rotations.shape[0], -1)
            batch_target = batch_rotations.reshape(batch_rotations.shape[0], -1)
            if torch.cuda.is_available():
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
            deep_latent, output = model(batch_input)
            errors.append(torch.norm(output - batch_target).item())
        print('##### Evaluation for Epoch %d, the reconstuction error is %04f' % (epoch_idx, np.mean(errors)))

    ### Store the results as bvh file
    model.eval()
    real_rotations, real_root_positions, outputs, file_names = [], [], [], []
    for batch_idx, (batch_rotations, _, batch_root_positions, file_name) in enumerate(test_dataloader):   
        batch_input = batch_rotations.reshape(batch_rotations.shape[0], -1)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        deep_latent, output = model(batch_input)
        outputs.append(output.cpu().detach().numpy() if torch.cuda.is_available() else output.detach().numpy())
        real_rotations.append(batch_input.cpu().detach().numpy() if torch.cuda.is_available() else batch_input.detach().numpy())
        real_root_positions.append(batch_root_positions.cpu().detach().numpy() if torch.cuda.is_available() else batch_root_positions.detach().numpy())
        file_names += file_name

    pre_rotations = np.concatenate(outputs, axis=0).reshape((-1, joint_number, rotation_number))
    real_rotations = np.concatenate(real_rotations, axis=0).reshape((-1, joint_number, rotation_number))
    real_root_positions = np.concatenate(real_root_positions, axis=0).reshape((-1, 3))
    file_names_set = set(file_names)
    for file_name in file_names_set:
        frame_indices = np.where(np.array(file_names)==file_name)[0]
        frame_number = frame_indices.shape[0]
        pre_rotation = Quaternions(pre_rotations[frame_indices]) if model_rotation_type == 'q' else Quaternions.from_euler(pre_rotations[frame_indices], order='zyx')
        real_rotation = Quaternions(real_rotations[frame_indices]) if model_rotation_type == 'q' else Quaternions.from_euler(real_rotations[frame_indices], order='zyx')
        positions = np.zeros((frame_number, joint_number, 3))
        positions[:, 0] = real_root_positions[frame_indices, 0]
        save('./output/autoencoder_fc/gt_%s' % file_name, real_rotation.normalized(), positions, train_dataset.offsets, train_dataset.parents, train_dataset.names, train_dataset.frametime*4)
        save('./output/autoencoder_fc/rec_%s' % file_name, pre_rotation.normalized(), positions, train_dataset.offsets, train_dataset.parents, train_dataset.names, train_dataset.frametime*4)
