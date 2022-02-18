import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from utils import *


def visualization(rotation, width):
    width = min(width, rotation.shape[0])
    data = rotation.reshape(rotation.shape[0], -1)
    data = data[:width]
    data = data.T
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cm.RdBu)
    plt.show()

bvh_file_path = './data/motion_walking.bvh'
rotations, positions, offsets, parents, names, frametime = load(filename=bvh_file_path)

rotations_eular = rotations.euler()

# visualization(rotations.qs, 1000)
# visualization(rotations_eular, 1000)
key_frames = np.arange(0, rotations.shape[0], 4)
visualization(rotations_eular[key_frames], 400)
