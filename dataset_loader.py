from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
import os
import re

# added
from torch.utils.data import Sampler
import random

class DrivingDataset(Dataset):

    def __init__(self, root_dir, categorical = False, classes=-1, transform=None):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in listdir(self.root_dir) if f.endswith('jpg')]
        self.categorical = categorical
        self.classes = classes

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        basename = self.filenames[idx]
        img_name = os.path.join(self.root_dir, basename)
        image = io.imread(img_name)

        m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
        steering_command = np.array(float(m.group(3)), dtype=np.float32)

        if self.categorical:
            steering_command = int(((steering_command + 1.0)/2.0) * (self.classes - 1))

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'cmd': steering_command}

    # Added helper function here
    def get_steering(self, idx):
        m = re.search(r'expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', self.filenames[idx])
        return float(m.group(3))

# Added batch sampler
class BalancedBatchSampler(Sampler):
    """50% straight 50% turning"""
    def __init__(self, dataset, batch_size, straight_thresh=0.1):
        self.batch_size = batch_size
        self.straight_idx, self.turn_idx = [], []
        for i in range(len(dataset)):
            # class 9, 10, 11
            if abs(dataset.get_steering(i)) < straight_thresh:
                self.straight_idx.append(i)
            else:
                self.turn_idx.append(i)
        self.n_batches = len(dataset) // batch_size

    def __iter__(self):
        half = self.batch_size // 2
        # Note:
        # random.choices(idx_list, k) => with replacement -> cause we don't have enought turn. Use this.
        # random.sample(idx_list, k) => without replacement
        # ex: turn_idx = [3, 8, 15] -> choice(turn_idx, k=5) -> [8, 3, 8, 15, 8]
        for _ in range(self.n_batches):
            batch = random.choices(self.straight_idx, k=half) + \
                    random.choices(self.turn_idx, k=self.batch_size - half)
            # [straight, straight, turn, turn] -> expected [s, t, s, t]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
