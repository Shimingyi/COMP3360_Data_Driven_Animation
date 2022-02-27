import torch
import torch.nn as nn

import os
import subprocess
import numpy as np

from utils import FK, load, save, Quaternions
from torch.utils.data import Dataset, DataLoader

## Training hyper parameters
epoch = 100
batch_size = 256
learning_rate = 1e-3

## Model parameters
model_rotation_type = 'q'

class MotionAutoEncoder_Conv(nn.Module):
    def __init__(self, input_feature_size):
        super(MotionAutoEncoder_Conv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_feature_size, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm1d(128), nn.ReLU(True), 
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=1),
            nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, stride=1),
            nn.BatchNorm1d(512), nn.ReLU(True), 
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=1),
            nn.BatchNorm1d(256), nn.ReLU(True), 
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=1),
            nn.BatchNorm1d(128), nn.ReLU(True), 
            nn.ConvTranspose1d(in_channels=128, out_channels=input_feature_size, kernel_size=5, stride=1),
            nn.ReLU(True), 
        )
        self.out_conv = nn.Conv1d(in_channels=input_feature_size, out_channels=input_feature_size, kernel_size=1)
    
    def forward(self, x):
        latent = self.encoder(x)
        # reconstrcution = self.decoder(latent)
        reconstrcution = self.out_conv(self.decoder(latent))
        return latent, reconstrcution


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
    model_clip_offset = 5
    train_dataset = MotionDataset('./data/edin_locomotion', model_rotation_type, model_clip_size, model_clip_offset, is_train=True)
    test_dataset = MotionDataset('./data/edin_locomotion_valid', model_rotation_type, model_clip_size, model_clip_size, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    joint_number, rotation_number = train_dataset.joint_number, train_dataset.rotation_number
    model = MotionAutoEncoder_Conv(input_feature_size=test_dataset.__get_feature_number__())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if torch.cuda.is_available():
        model.cuda()

    for epoch_idx in range(epoch):
        print_freq = len(train_dataloader) // 10

        model.train()
        for batch_idx, (batch_rotations, _, batch_root_positions, file_name) in enumerate(train_dataloader):   
            batch_input = batch_rotations.reshape((batch_rotations.shape[0], batch_rotations.shape[1], -1)).transpose(1, 2)
            batch_target = batch_rotations.reshape((batch_rotations.shape[0], batch_rotations.shape[1], -1)).transpose(1, 2)

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
            batch_input = batch_rotations.reshape((batch_rotations.shape[0], batch_rotations.shape[1], -1)).transpose(1, 2)
            batch_target = batch_rotations.reshape((batch_rotations.shape[0], batch_rotations.shape[1], -1)).transpose(1, 2)
            if torch.cuda.is_available():
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
            deep_latent, output = model(batch_input)
            errors.append(torch.norm(output - batch_target).item())
        print('##### Evaluation for Epoch %d, the reconstuction error is %04f' % (epoch_idx, np.mean(errors)))

    model.eval()
    real_rotations, real_root_positions, outputs, file_names = [], [], [], []
    for batch_idx, (batch_rotations, _, batch_root_positions, file_name) in enumerate(test_dataloader):   
        batch_input = batch_rotations.reshape((batch_rotations.shape[0], batch_rotations.shape[1], -1)).transpose(1, 2)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        deep_latent, output = model(batch_input)
        outputs.append(output.transpose(1, 2).cpu().detach().numpy() if torch.cuda.is_available() else output.detach().numpy())
        real_rotations.append(batch_input.transpose(1, 2).cpu().detach().numpy() if torch.cuda.is_available() else batch_input.detach().numpy())
        real_root_positions.append(batch_root_positions.cpu().detach().numpy() if torch.cuda.is_available() else batch_root_positions.detach().numpy())
        file_names += file_name*model_clip_size

    pre_rotations = np.concatenate(outputs, axis=0).reshape((-1, joint_number, rotation_number))
    real_rotations = np.concatenate(real_rotations, axis=0).reshape((-1, joint_number, rotation_number))
    real_root_positions = np.concatenate(real_root_positions, axis=0).reshape((-1, 3))
    file_names_set = set(file_names)
    pre_rotations[:, 0] = real_rotations[:, 0]
    for file_name in file_names_set:
        frame_indices = np.where(np.array(file_names)==file_name)[0]
        frame_number = frame_indices.shape[0]
        pre_rotation = Quaternions(pre_rotations[frame_indices]) if model_rotation_type is 'q' else Quaternions.from_euler(pre_rotations[frame_indices], order='zyx')
        real_rotation = Quaternions(real_rotations[frame_indices]) if model_rotation_type is 'q' else Quaternions.from_euler(real_rotations[frame_indices], order='zyx')
        positions = np.zeros((frame_number, joint_number, 3))
        positions[:, 0] = real_root_positions[frame_indices]
        save('./output/autoencoder_conv1d/gt_%s' % file_name, real_rotation.normalized(), positions, train_dataset.offsets, train_dataset.parents, train_dataset.names, train_dataset.frametime*4)
        save('./output/autoencoder_conv1d/rec_%s' % file_name, pre_rotation.normalized(), positions, train_dataset.offsets, train_dataset.parents, train_dataset.names, train_dataset.frametime*4)
