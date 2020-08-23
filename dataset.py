import os
import random
import torch
from torch.utils.data import Dataset
import pickle
import collections

import numpy as np


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', phase = "train", crop_size=[75, 75, 75], num_input=5):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line , name + '_')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.phase = phase
        self.random_flip = True
        self.num_input = num_input
        self.crop_size = crop_size

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, mask = pkload(path + 'data_f32.pkl')
        y[y == 4] = 3
        # print(x.shape, y.shape)#(240, 240, 155, 4) (240, 240, 155)
        mask = mask.astype(np.int)
        # x, y, mask = x[None, ...], y[None, ...], mask[None, ...]
        x = x.transpose(3, 0, 1, 2)
        # print(x.shape, y.shape, mask.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
        # x, y, mask = self.transforms([x, y, mask])
        sample = {'images': x, 'mask': mask, 'labels':y}
        if self.phase == "train":
            transform = RandomCrop(self.crop_size, self.random_flip, self.num_input)
            sample = transform(sample)
        else:
            sample["images"] = torch.tensor(sample["images"], dtype=torch.float)
            sample["labels"] = torch.tensor(sample["labels"], dtype=torch.float)
            sample["mask"] = torch.tensor(sample["mask"], dtype=torch.float)
            sample["name"] = self.names[index]

        return sample

    def __len__(self):
        return len(self.names)

class RandomCrop(object):
    def __init__(self, output_size, random_flip, num_input):
        assert len(output_size) == 3
        self.output_size = output_size        
        self.random_flip = random_flip
        self.num_input = num_input
    
    def __call__(self, sample):
        images, labels, mask = sample['images'], sample['labels'], sample['mask']   
        h, w, d = self.output_size
       
        # generate the training batch with equal probability for the foreground and background
        # within the mask
        labelm = labels + mask
        # choose foreground or background
        fb = np.random.choice(2)
        if fb:
            index = np.argwhere(labelm > 1)
        else:
            index = np.argwhere(labelm == 1)
        # choose the center position of the image segments
        choose = random.sample(range(0, len(index)), 1)
        center = index[choose].astype(int)
        center = center[0]
        
        # check whether the left and right index overflow
        left = []
        for i in range(3):
        	margin_left = int(self.output_size[i]/2)
        	margin_right = self.output_size[i] - margin_left
        	left_index = center[i] - margin_left
        	right_index = center[i] + margin_right
        	if left_index < 0:
        		left_index = 0
        	if right_index > labels.shape[i]:
        		left_index = left_index - (right_index - labels.shape[i])
        	left.append(left_index)
        	
        # crop the image and the label to generate image segments
        image = np.zeros([self.num_input - 1, h, w, d])
        label = np.zeros([h, w, d])
        
        image = images[:, left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]
        label = labels[left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]        
        
        # random flip 
        if self.random_flip:       
        	flip = np.random.choice(2)*2-1
        	image = image[:,::flip,:,:]
        	label = label[::flip,:,:]

        return {'images':torch.tensor(image.copy(), dtype=torch.float), 'labels': torch.tensor(label.copy(), dtype=torch.long)}




if __name__ == "__main__":
    train_list = os.path.join("/home/koga/dataset/BRATS2018/Train", "train_0.txt")
    dataset = BraTSDataset(list_file = train_list, root="/home/koga/dataset/BRATS2018/Train")